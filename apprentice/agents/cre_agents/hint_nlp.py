"""
Lightweight hint interpreter for demos.

Given a state (dict or ProblemState), a demo Action, and a natural-language
hint, we extract simple preconditions and emit executable CRE code such as
`conds = AND(angle_ab.value == 90, a.value != "", b.value != "")`.

This is intentionally rule-based/cheap to keep it dependency-free. It can be
swapped for a heavier NLP model later; the interface stays the same.
"""

from __future__ import annotations

import re
import os
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any, Callable, Optional, List
import ast
import json

from tutorgym.shared import Action

try:
    # Fact types for generating executable CRE code.
    from apprentice.agents.cre_agents.environment import TextField, Label, Button, Component
    _FACT_TYPES_BY_NAME = {
        "TextField": TextField,
        "Label": Label,
        "Button": Button,
        "Component": Component,
    }
except Exception:
    TextField = Label = Button = Component = None
    _FACT_TYPES_BY_NAME = {}


def _as_objs(state: Any) -> Dict[str, Dict[str, Any]]:
    """Accept either a ProblemState or a plain dict of field objects."""
    if hasattr(state, "objs"):
        return state.objs  # type: ignore[attr-defined]
    if isinstance(state, dict):
        return state
    raise TypeError(f"Unsupported state type: {type(state)}")


def _numeric_tokens(hint: str) -> Iterable[str]:
    """Yield numeric tokens (ints/floats) mentioned in the hint."""
    for tok in re.findall(r"\d+(?:\.\d+)?", hint):
        yield tok


def _obj_type_name(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        return obj.get("type")
    return getattr(obj, "type", None)


def _infer_fact_type_name(var_name: str, objs: Optional[Dict[str, Dict[str, Any]]]) -> str:
    if objs and var_name in objs:
        type_name = _obj_type_name(objs[var_name])
        if type_name:
            return type_name
    if var_name.startswith("label_of_"):
        return "Label"
    if var_name == "done":
        return "Button"
    return "TextField"


def _normalize_rhs(rhs: str) -> str:
    rhs = rhs.strip()
    if rhs.lower() in ("true", "false", "none"):
        return rhs.title() if rhs.lower() != "none" else "None"
    if (rhs.startswith("'") and rhs.endswith("'")) or (rhs.startswith('"') and rhs.endswith('"')):
        return rhs
    if re.match(r"^-?\d+(\.\d+)?$", rhs):
        return rhs
    return f"\"{rhs}\""


_TERM_RE = re.compile(r"^\s*(?P<left>[A-Za-z0-9_\.]+)\s*(?P<op>==|!=|>=|<=|>|<)\s*(?P<right>.+?)\s*$")


def _expr_to_cre_code(expr: str, objs: Optional[Dict[str, Dict[str, Any]]]) -> str:
    expr = expr.strip()
    if not expr:
        return ""
    if expr.lower() in ("true", "false"):
        return expr

    or_groups = [g.strip() for g in expr.split("||")] if "||" in expr else [expr]
    var_names: List[str] = []
    group_exprs: List[str] = []

    for group in or_groups:
        and_terms = [t.strip() for t in group.split("&&")] if "&&" in group else [group]
        term_exprs: List[str] = []
        for raw_term in and_terms:
            term = raw_term.strip().strip("()")
            if not term:
                continue
            match = _TERM_RE.match(term)
            if match:
                left = match.group("left").strip()
                op = match.group("op").strip()
                right = _normalize_rhs(match.group("right"))
                if "." in left:
                    var_name, attr = left.split(".", 1)
                else:
                    var_name, attr = left, "value"
                var_names.append(var_name)
                left_expr = f"{var_name}.{attr}" if attr else var_name
                term_exprs.append(f"{left_expr} {op} {right}")
            else:
                var_name = term
                var_names.append(var_name)
                term_exprs.append(var_name)
        if term_exprs:
            group_exprs.append(f"AND({', '.join(term_exprs)})")

    if not group_exprs:
        return ""

    if len(group_exprs) == 1:
        conds_expr = group_exprs[0]
    else:
        conds_expr = f"OR({', '.join(group_exprs)})"

    # Build Var declarations with inferred fact types.
    type_names = []
    seen_vars = []
    for var_name in var_names:
        if var_name in seen_vars:
            continue
        seen_vars.append(var_name)
        type_name = _infer_fact_type_name(var_name, objs)
        if type_name not in type_names:
            type_names.append(type_name)

    # Stable import order with any extra types appended.
    base_order = ["TextField", "Label", "Button", "Component"]
    ordered_types = [t for t in base_order if t in type_names]
    extras = [t for t in type_names if t not in base_order]
    ordered_types.extend(sorted(extras))
    if not ordered_types:
        ordered_types = ["TextField"]

    imports = [
        "from cre import Var",
        "from cre.conditions import AND, OR",
        f"from apprentice.agents.cre_agents.environment import {', '.join(ordered_types)}",
        "",
    ]

    var_lines = []
    for var_name in seen_vars:
        type_name = _infer_fact_type_name(var_name, objs)
        var_lines.append(f"{var_name} = Var({type_name}, '{var_name}')")

    code_lines = imports + var_lines + [f"conds = {conds_expr}"]
    return "\n".join(code_lines)


def _ensure_conds_var(code: str) -> str:
    if re.search(r"^\s*conds\s*=", code, flags=re.MULTILINE):
        return code
    lines = [line for line in code.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return code
    candidate = None
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith(("AND(", "OR(", "NOT(")) and "conds" not in stripped:
            candidate = stripped
            break
    if candidate:
        return code + f"\nconds = {candidate}"
    return code


def _normalize_cre_code(response: str, state: Any) -> str:
    response = response.strip()
    if not response:
        return response
    if "llm_error" in response:
        return response
    if "Var(" in response or "conds" in response or response.startswith("from "):
        return _ensure_conds_var(response)

    try:
        if isinstance(state, dict) and "_objs" in state:
            objs = state.get("_objs")
        else:
            objs = _as_objs(state)
    except Exception:
        objs = None
    return _expr_to_cre_code(response, objs)


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_TERM_IN_LINE_RE = re.compile(r"([A-Za-z0-9_]+\.[A-Za-z0-9_]+)\s*(==|!=|>=|<=|>|<)\s*([^,\)\n]+)")


def _extract_code_block(text: str) -> str:
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return max((m.strip() for m in matches), key=len, default="").strip()
    return text.strip()


def _line_looks_like_code(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if stripped.startswith(("from ", "import ", "conds", "AND(", "OR(", "NOT(")):
        return True
    if "Var(" in stripped or ".locked" in stripped or ".value" in stripped:
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*Var\(", stripped):
        return True
    return False


def _strip_non_code_lines(text: str) -> str:
    lines = [line for line in text.splitlines() if _line_looks_like_code(line)]
    return "\n".join(lines).strip()


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _has_conds_assignment(code: str) -> bool:
    return bool(re.search(r"^\s*conds\s*=", code, flags=re.MULTILINE))


def _extract_predicate_expr(text: str) -> str:
    terms: List[str] = []
    for line in text.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        stripped = re.sub(r"^(AND|OR|NOT|conds)\s*=?\s*\(?", "", stripped).strip()
        stripped = stripped.rstrip(")")
        stripped = re.sub(r"(?:Label|TextField|Button|Component)\('([^']+)'\)\.(\w+)", r"\1.\2", stripped)
        match = _TERM_IN_LINE_RE.search(stripped)
        if match:
            left, op, right = match.groups()
            terms.append(f"{left} {op} {right.strip()}")
            continue
        if ".locked" in stripped and "==" not in stripped and "!=" not in stripped:
            terms.append(f"{stripped} == True")
        elif ".value" in stripped and "==" not in stripped and "!=" not in stripped:
            terms.append(f"{stripped} != \"\"")
    return " && ".join(terms)


def _sanitize_llm_code(response: str, state: Any) -> str:
    if not response:
        return response
    code = _extract_code_block(response)
    code = _strip_non_code_lines(code)
    if code:
        code = _ensure_conds_var(code)
        if _is_valid_python(code) and _has_conds_assignment(code):
            return code
        # Try truncating to a valid prefix.
        lines = code.splitlines()
        for i in range(len(lines), 0, -1):
            candidate = "\n".join(lines[:i]).strip()
            if not candidate:
                continue
            candidate = _ensure_conds_var(candidate)
            if _is_valid_python(candidate) and _has_conds_assignment(candidate):
                return candidate

    try:
        if isinstance(state, dict) and "_objs" in state:
            objs = state.get("_objs")
        else:
            objs = _as_objs(state)
    except Exception:
        objs = None
    expr = _extract_predicate_expr(response)
    if expr:
        return _expr_to_cre_code(expr, objs)
    return ""


def interpret_hint(state, demo: Action, hint: str) -> str:
    """
    Return executable CRE code for likely preconditions implied by `hint`.

    Heuristics:
    - Match numeric mentions in the hint to field values.
    - If the hint mentions "angle" and the field name contains "angle",
      include exact-value matches.
    - Always include non-empty inputs of the demo selection as required.

    Parameters
    ----------
    state : ProblemState | dict
    demo : Action
    hint : str
    """
    objs = _as_objs(state)
    hint_lower = hint.lower()
    numbers = set(_numeric_tokens(hint_lower))

    clauses = []

    # Anchor on the demo selection: require its inputs to be non-empty.
    sel = demo.selection if isinstance(demo, Action) else None
    if sel and sel in objs:
        clauses.append(f"{sel} != ''")

    # Match numeric hints to fields; prefer angle fields if "angle" is mentioned.
    for field, obj in objs.items():
        val = obj.get("value", "")
        if val is None or val == "":
            continue

        # Basic non-empty requirement for core triangle inputs.
        if field in ("a", "b", "c") or field.startswith(("side_", "angle_")):
            clauses.append(f"{field} != ''")

        # Exact numeric match from hint.
        if str(val) in numbers:
            clauses.append(f"{field}.value=={val}")
            continue

        # Angle cues: hint mentions angle and field name contains angle.
        if "angle" in hint_lower and "angle" in field and val != "":
            clauses.append(f"{field}.value=={val}")

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for c in clauses:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    try:
        if isinstance(state, dict) and "_objs" in state:
            objs = state.get("_objs")
        else:
            objs = _as_objs(state)
    except Exception:
        objs = None
    return _expr_to_cre_code(" && ".join(deduped), objs)


# ------------------------------
# Optional: LLM-backed interpreter
# ------------------------------

def get_openai_token() -> Optional[str]:
    """
    Retrieve the OpenAI API key from environment variables or a secrets file.
    """
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    
    secrets_paths = [Path("secrets.json")]
    for path in secrets_paths:
        if path.exists():
            try:
                import json
                with open(path, "r") as f:
                    data = json.load(f)
                    return data.get("OPENAI_API_KEY")
            except Exception:
                continue
    return None


def get_openai_base_url() -> Optional[str]:
    """
    Retrieve an optional OpenAI-compatible base URL from environment variables
    or a local secrets file.
    """
    for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw

    secrets_paths = [Path("secrets.json")]
    for path in secrets_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
                        raw = str(data.get(key, "")).strip()
                        if raw:
                            return raw
            except Exception:
                continue
    return None


def get_openai_model(default: str = "gpt-4-turbo") -> str:
    """
    Retrieve the hint-LLM model name from environment variables or a secrets file.
    """
    for key in ("CRE_HINT_OPENAI_MODEL", "OPENAI_MODEL"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw

    secrets_paths = [Path("secrets.json")]
    for path in secrets_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    for key in ("CRE_HINT_OPENAI_MODEL", "OPENAI_MODEL"):
                        raw = str(data.get(key, "")).strip()
                        if raw:
                            return raw
            except Exception:
                continue
    return default


def _get_llm_seed() -> Optional[int]:
    raw = os.environ.get("CRE_HINT_LLM_SEED", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_llm_temperature(default: float = 0.0) -> float:
    raw = os.environ.get("CRE_HINT_LLM_TEMPERATURE", "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_llm_cache_dir() -> Optional[Path]:
    raw = os.environ.get("CRE_HINT_LLM_CACHE_DIR", "").strip()
    if not raw:
        return None
    cache_dir = Path(raw).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_key(model_name: str, messages: List[Dict[str, str]], seed: Optional[int], temperature: float) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "seed": seed,
        "temperature": temperature,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_openai_llm_call(model_name: str = "gpt-4-turbo"):
    """
    Builds a callable that uses the OpenAI Chat Completion API.
    """
    api_key = get_openai_token()
    if not api_key:
        return None

    try:
        from openai import OpenAI
        base_url = get_openai_base_url()
        resolved_model_name = get_openai_model(model_name)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        seed = _get_llm_seed()
        temperature = _get_llm_temperature(0.0)
        cache_dir = _get_llm_cache_dir()
        
        def _call(msgs_or_str):
            if isinstance(msgs_or_str, str):
                messages = [{"role": "user", "content": msgs_or_str}]
            else:
                messages = msgs_or_str

            cache_path = None
            if cache_dir is not None:
                key = _cache_key(resolved_model_name, messages, seed, temperature)
                cache_path = cache_dir / f"{key}.json"
                if cache_path.exists():
                    try:
                        cached = json.loads(cache_path.read_text(encoding="utf-8"))
                        return str(cached.get("response", ""))
                    except Exception as exc:
                        print(f"[hint_nlp] Cache read failed for {cache_path}: {exc}")
                
            try:
                request_kwargs = {
                    "model": resolved_model_name,
                    "messages": messages,
                    "temperature": temperature,
                }
                if seed is not None:
                    request_kwargs["seed"] = seed
                completion = client.chat.completions.create(
                    **request_kwargs
                )
                response = completion.choices[0].message.content.strip()
                if cache_path is not None:
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_payload = {
                            "model": resolved_model_name,
                            "base_url": base_url or "",
                            "seed": seed,
                            "temperature": temperature,
                            "messages": messages,
                            "response": response,
                        }
                        cache_text = json.dumps(
                            cache_payload,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                        cache_path.write_text(cache_text, encoding="utf-8")
                    except Exception as exc:
                        print(f"[hint_nlp] Cache write failed for {cache_path}: {exc}")
                return response
            except Exception as exc:
                return f"#llm_error {exc}"
        
        if base_url:
            print(f"[hint_nlp] Using OpenAI-compatible API with model: {resolved_model_name} @ {base_url}")
        else:
            print(f"[hint_nlp] Using OpenAI API with model: {resolved_model_name}")
        return _call
    except ImportError:
        print("[hint_nlp] OpenAI python package not installed.")
        return None


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert at translating natural language hints into CRE (Cognitive Rule Engine) Condition objects.\n\n"
    "Generate Python code that constructs a CRE Conditions object for the given hint.\n\n"
    "Output Format:\n"
    "1. Imports:\n"
    "   - `from cre import Var`\n"
    "   - `from cre.conditions import AND, OR`\n"
    "   - `from apprentice.agents.cre_agents.environment import TextField, Label, Button, Component`\n"
    "2. Create Vars with aliases:\n"
    "   - `field = Var(TextField, 'field')`\n"
    "3. Build conditions using comparisons on `.value` or `.locked`:\n"
    "   - `conds = AND(field.value == \"\", other.value != \"\")`\n"
    "4. Final variable must be named `conds`.\n\n"
    "Use Label for ids like `label_of_*`. Prefer the input field (e.g., `apply_pythagorean`) over its label unless the hint explicitly mentions the label text.\n"
    "Output ONLY valid Python code."
)


def _serialize_state_for_prompt(objs: Dict[str, Dict[str, Any]], limit: int = 200) -> str:
    parts = []
    for idx, (k, v) in enumerate(objs.items()):
        if idx >= limit:
            break
        val = v.get("value", "")
        obj_type = _obj_type_name(v)
        type_suffix = f" ({obj_type})" if obj_type else ""
        parts.append(f'{k}{type_suffix}: "{val}"')
    return "\n".join(parts)


def _serialize_predicates(preds: Iterable[Any]) -> str:
    return "\n".join([f"- {p}" for p in preds])


def _prep_prompt(state, demo: Action, hint: str) -> str:
    if isinstance(state, dict) and "_predicates" in state and "_objs" in state:
        preds = state.get("_predicates") or []
        objs = state.get("_objs") or {}
        trimmed = list(preds)[:100]
        state_block = _serialize_state_for_prompt(objs)
        return (
            "Predicates (one per line):\n"
            f"{_serialize_predicates(trimmed)}\n\n"
            "State fields:\n"
            f"{state_block}\n\n"
            f"Demo action: {demo}\n"
            f"Hint: {hint}\n"
            "CRE Code:"
        )
    if hasattr(state, "get_facts"):
        preds = [str(fact) for fact in state.get_facts()]
        trimmed = preds[:100]
        return (
            "Predicates (one per line):\n"
            f"{_serialize_predicates(trimmed)}\n\n"
            f"Demo action: {demo}\n"
            f"Hint: {hint}\n"
            "CRE Code:"
        )
    # If callers pass a featurized predicate list/array, present it directly.
    if isinstance(state, (list, tuple)):
        trimmed = list(state)[:100]  # cap prompt size
        return (
            "Predicates (one per line):\n"
            f"{_serialize_predicates(trimmed)}\n\n"
            f"Demo action: {demo}\n"
            f"Hint: {hint}\n"
            "Precondition:"
        )
    try:
        objs = _as_objs(state)
        state_block = _serialize_state_for_prompt(objs)
    except Exception:
        state_block = str(state)

    return (
        "Example:\n"
        "State:\nnormalize_inputs (TextField): \"\"\nclassify_triangle (TextField): \"\"\n"
        "Demo action: input(normalize_inputs, OK)\n"
        "Hint: Normalize all triangle inputs.\n"
        "CRE Code:\n"
        "```python\n"
        "from cre import Var\n"
        "from cre.conditions import AND, OR\n"
        "from apprentice.agents.cre_agents.environment import TextField, Label, Button, Component\n"
        "normalize_inputs = Var(TextField, 'normalize_inputs')\n"
        "conds = AND(normalize_inputs.value == \"\")\n"
        "```\n\n"
        f"State:\n{state_block}\n\n"
        f"Demo action: {demo}\n"
        f"Hint: {hint}\n"
        "CRE Code:"
    )


def interpret_hint_with_llm(state, demo: Action, hint: str, llm_call) -> str:
    """
    Use an external language model to map (state, demo, hint) -> precondition string.

    `llm_call` can accept either a list of chat messages or a plain prompt string.
    For featurized states (list of predicates), the prompt highlights only those predicates.
    """
    system = {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    user = {"role": "user", "content": _prep_prompt(state, demo, hint)}
    
    try:
        response = llm_call([system, user])
    except TypeError:
        response = llm_call(user["content"])
    
    # Basic cleanup of the response.
    response = response.strip()
    if response.startswith("```"):
        response = response.strip("`").strip()
        if response.startswith("python"):
            response = response[6:].strip()

    # Normalize and sanitize to executable CRE code.
    code = _sanitize_llm_code(response, state)
    if not code:
        strict_prompt = (
            _prep_prompt(state, demo, hint)
            + "\n\nReturn ONLY valid Python code. No markdown, no comments. "
              "Must include a final line like: conds = AND(...)."
        )
        strict_user = {"role": "user", "content": strict_prompt}
        try:
            retry = llm_call([system, strict_user])
        except TypeError:
            retry = llm_call(strict_user["content"])
        code = _sanitize_llm_code(str(retry), state)

    if not code:
        fallback_state = state.get("_objs") if isinstance(state, dict) and "_objs" in state else state
        return interpret_hint(fallback_state, demo, hint)
    return code


def get_hf_token() -> Optional[str]:
    """
    Retrieve the Hugging Face token from environment variables or a secrets file.
    """
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    
    # Check for secrets file in standard locations or locally
    secrets_paths = [
        Path("secrets.json"),
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token"
    ]
    
    for path in secrets_paths:
        if path.exists():
            try:
                if path.suffix == ".json":
                    import json
                    with open(path, "r") as f:
                        data = json.load(f)
                        return data.get("HF_TOKEN")
                else:
                    with open(path, "r") as f:
                        return f.read().strip()
            except Exception:
                continue
    return None


def _force_local_llm() -> bool:
    raw = os.environ.get("CRE_HINT_LLM_FORCE_LOCAL", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def build_hf_llm_call(model_name_or_path: str = "distilgpt2", max_new_tokens: int = 64, max_input_tokens: int = 512):
    """
    Build a Hugging Face text-generation callable.
    Checks for OpenAI key first, then HF token, then falls back to local.
    """
    force_local = _force_local_llm()
    if not force_local:
        # 1. Check OpenAI
        openai_call = build_openai_llm_call()
        if openai_call:
            return openai_call

    # 2. Check HF API
    token = None if force_local else get_hf_token()
    
    # --- API Client Path ---
    if token:
        try:
            from huggingface_hub import InferenceClient
            # Use a better model by default if using the API
            api_model = "meta-llama/Meta-Llama-3-8B-Instruct" 
            client = InferenceClient(model=api_model, token=token)
            
            def _api_call(msgs_or_str):
                # Format prompt for chat models if needed
                if isinstance(msgs_or_str, str):
                     messages = [{"role": "user", "content": msgs_or_str}]
                else:
                    messages = msgs_or_str
                
                try:
                    # We can use chat completion for better instruction following
                    out = client.chat_completion(
                        messages, 
                        max_tokens=max_new_tokens,
                        temperature=0.7
                    )
                    return out.choices[0].message.content.strip()
                except Exception as exc:
                     return f"#llm_error {exc}"

            print(f"[hint_nlp] Using HF Inference API with model: {api_model}")
            return _api_call
            
        except ImportError:
            print("[hint_nlp] huggingface_hub not installed or failed. Falling back to local.")
        except Exception as e:
             print(f"[hint_nlp] API client init failed: {e}. Falling back to local.")

    # --- Local Pipeline Path ---
    try:
        from transformers import AutoTokenizer, pipeline, set_seed
    except ImportError as e:
        error_msg = str(e)
        def _missing(_):
            raise RuntimeError(f"transformers not installed; provide your own llm_call. Error: {error_msg}")
        return _missing

    seed = _get_llm_seed()
    temperature = _get_llm_temperature(0.7)
    cache_dir = _get_llm_cache_dir()

    try:
        print(f"[hint_nlp] Loading local model: {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        generator = pipeline(
            "text-generation",
            model=model_name_or_path,
            tokenizer=tokenizer,
            # device=0 if torch.cuda.is_available() else -1  # usage depends on environment
        )
    except Exception as e:
        error_msg = str(e)
        def _missing(_):
            raise RuntimeError(f"Failed to load model '{model_name_or_path}': {error_msg}")
        return _missing

    def _truncate(prompt: str) -> str:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= max_input_tokens:
            return prompt
        trimmed = ids[:max_input_tokens]
        return tokenizer.decode(trimmed, clean_up_tokenization_spaces=True)

    def _call(msgs_or_str):
        if isinstance(msgs_or_str, list):
            # Naively join for base models
            messages = msgs_or_str
            prompt = "\n\n".join(m.get("content", "") for m in msgs_or_str if m.get("content"))
        else:
            prompt = str(msgs_or_str)
            messages = [{"role": "user", "content": prompt}]
        prompt = _truncate(prompt)
        cache_path = None
        if cache_dir is not None:
            key = _cache_key(
                f"local::{model_name_or_path}",
                [{"role": "user", "content": prompt}],
                seed,
                temperature,
            )
            cache_path = cache_dir / f"{key}.json"
            if cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    return str(cached.get("response", ""))
                except Exception as exc:
                    print(f"[hint_nlp] Cache read failed for {cache_path}: {exc}")
        try:
            if seed is not None:
                set_seed(seed)
            # Adjust generation parameters for better results
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                generate_kwargs["temperature"] = temperature
            out = generator(prompt, **generate_kwargs)[0]["generated_text"]
            response = out[len(prompt):].strip() or out.strip()
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_payload = {
                        "model": f"local::{model_name_or_path}",
                        "seed": seed,
                        "temperature": temperature,
                        "messages": messages,
                        "truncated_prompt": prompt,
                        "response": response,
                    }
                    cache_text = json.dumps(
                        cache_payload,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                    cache_path.write_text(cache_text, encoding="utf-8")
                except Exception as exc:
                    print(f"[hint_nlp] Cache write failed for {cache_path}: {exc}")
            return response
        except Exception as exc:
            return f"#llm_error {exc}"

    return _call
