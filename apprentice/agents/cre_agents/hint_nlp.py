"""
Lightweight hint interpreter for demos.

Given a state (dict or ProblemState), a demo Action, and a natural-language
hint, we extract simple, human-readable preconditions such as
`angle_ab.value == 90 && a != '' && b != ''`.

This is intentionally rule-based/cheap to keep it dependency-free. It can be
swapped for a heavier NLP model later; the interface stays the same.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple, Any, Callable, Optional

from tutorgym.shared import Action


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


def interpret_hint(state, demo: Action, hint: str) -> str:
    """
    Return a conjunctive string of likely preconditions implied by `hint`.

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

    return " && ".join(deduped)


# ------------------------------
# Optional: LLM-backed interpreter
# ------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert at translating natural language hints into logical preconditions for a cognitive agent. "
    "Given a problem state (a list of fields with values) and a hint, your task is to output a boolean expression "
    "that represents the condition described in the hint. "
    "Use the format `field.value == value` or `field != ''`. "
    "Combine conditions with `&&`. "
    "Do not include any explanation, just the boolean expression."
)


def _serialize_state_for_prompt(objs: Dict[str, Dict[str, Any]]) -> str:
    parts = []
    for k, v in objs.items():
        val = v.get("value", "")
        # locked = v.get("locked", False)
        # Simplify state representation for the LLM
        if val != "":
            parts.append(f'{k}: "{val}"')
    return "\n".join(parts)


def _serialize_predicates(preds: Iterable[Any]) -> str:
    return "\n".join([f"- {p}" for p in preds])


def _prep_prompt(state, demo: Action, hint: str) -> str:
    # If callers pass a featurized predicate list/array, present it directly.
    if isinstance(state, (list, tuple)):
        trimmed = list(state)[:100]  # cap prompt size
        return (
            "Predicates (one per line):\n"
            f"{_serialize_predicates(trimmed)}\n\n"
            f"Demo action: {demo}\n"
            f"Hint: {hint}\n"
            "Select the predicates that the hint implies and combine them as a boolean precondition."
        )
    try:
        objs = _as_objs(state)
        state_block = _serialize_state_for_prompt(objs)
    except Exception:
        state_block = str(state)

    return (
        "Examples:\n"
        "State:\nangle_C: \"90\"\na: \"3\"\nb: \"4\"\n"
        "Demo action: input(c, 5)\n"
        "Hint: Use the Pythagorean theorem for right triangles.\n"
        "Precondition: angle_C.value == 90 && a != '' && b != ''\n\n"
        f"State:\n{state_block}\n\n"
        f"Demo action: {demo}\n"
        f"Hint: {hint}\n"
        "Precondition:"
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
    
    # Basic cleanup of the response
    response = response.strip()
    if response.startswith("```"):
        response = response.strip("`").strip()
    if response.lower().startswith("precondition:"):
        response = response[13:].strip()
    
    # Take only the first line to avoid hallucinated continuations
    if "\n" in response:
        response = response.split("\n")[0].strip()
        
    return response


def build_hf_llm_call(model_name_or_path: str = "distilgpt2", max_new_tokens: int = 64, max_input_tokens: int = 512):
    """
    Build a Hugging Face text-generation callable. Uses a tiny model by default.
    If transformers or the model is unavailable, raises a RuntimeError so callers
    can handle gracefully.
    """
    try:
        from transformers import pipeline, AutoTokenizer
    except ImportError as exc:
        def _missing(_):
            raise RuntimeError("transformers not installed; provide your own llm_call.") from exc
        return _missing

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        generator = pipeline(
            "text-generation",
            model=model_name_or_path,
            tokenizer=tokenizer,
        )
    except Exception as exc:
        def _missing(_):
            raise RuntimeError(f"Failed to load model '{model_name_or_path}': {exc}") from exc
        return _missing

    def _truncate(prompt: str) -> str:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= max_input_tokens:
            return prompt
        trimmed = ids[:max_input_tokens]
        return tokenizer.decode(trimmed, clean_up_tokenization_spaces=True)

    def _call(msgs_or_str):
        if isinstance(msgs_or_str, list):
            prompt = "\n\n".join(m.get("content", "") for m in msgs_or_str if m.get("content"))
        else:
            prompt = str(msgs_or_str)
        prompt = _truncate(prompt)
        try:
            # Adjust generation parameters for better results
            out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)[0]["generated_text"]
            return out[len(prompt):].strip() or out.strip()
        except Exception as exc:
            return f"#llm_error {exc}"

    return _call
