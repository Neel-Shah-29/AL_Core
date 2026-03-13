from numba.types import f8, string, boolean
from apprentice.agents.cre_agents.extending import registries, new_register_decorator, new_register_all
from apprentice.agents.cre_agents.environment import TextField
from cre import CREFunc
import numpy as np
import sympy as sp
from sympy import sstr, latex, symbols
import re
from numba import njit
ERROR_SENTINEL = "ERROR"
register_func = new_register_decorator("func", full_descr="CREFunc")
register_all_funcs = new_register_all("func", types=[CREFunc], full_descr="CREFunc")

@CREFunc(signature=boolean(string,string),
    shorthand = '{0} == {1}',
    commutes=True)
def Equals(a, b):
    return a == b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} + {1}',
    commutes=True)
def Add(a, b):
    return a + b

@CREFunc(signature=f8(f8,f8))
def AddPositive(a, b):
    if(not (a >= 0 and b >= 0)):
        raise Exception
    return a + b

@CREFunc(signature=f8(f8,f8,f8),
    shorthand = '{0} + {1} + {2}',
    commutes=True)
def Add3(a, b, c):
    return a + b + c

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} - {1}')
def Subtract(a, b):
    return a - b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} * {1}',
    commutes=True)
def Multiply(a, b):
    return a * b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} / {1}'
    )
def Divide(a, b):
    return a / b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} // {1}')
def FloorDivide(a, b):
    return a // b

# @CREFunc(signature=f8(f8,f8),
#     shorthand = '{0} ** {1}')
# def Power(a, b):
#     return a ** b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} % {1}')
def Modulus(a, b):
    return a % b


@CREFunc(signature=f8(f8), shorthand = '{0}^2')
def Square(a):
    return a * a

@CREFunc(signature=f8(f8, f8), shorthand = '{0}^{1}')
def Power(a, b):
    return a ** b

@CREFunc(signature=f8(f8), shorthand = '{0}+1')
def Increment(a):
    return a + 1

@CREFunc(signature=f8(f8), shorthand = '{0}-1')
def Decrement(a):
    return a - 1

@CREFunc(signature=f8(f8), shorthand = 'log2({0})')
def Log2(a):
    return np.log2(a)

@CREFunc(signature=f8(f8), shorthand = 'cos({0})')
def Cos(a):
    return np.cos(a)

@CREFunc(signature=f8(f8), shorthand = 'sin({0})')
def Sin(a):
    return np.sin(a)

@CREFunc(signature=f8(f8),
    shorthand = '{0} % 10')
def Mod10(a):
    return a % 10

@CREFunc(signature=f8(f8),
    shorthand = '{0} // 10')
def Div10(a):
    return a // 10

@CREFunc(signature=string(string),
    shorthand = '{0}')
def Copy(a):
    return a

@CREFunc(signature = string(string,string),
    shorthand = '{0} + {1}', 
    commutes=False)
def Concatenate(a, b):
    return a + b


@CREFunc(signature=f8(f8), shorthand = '{0}/2')
def Half(a):
    return a / 2

@CREFunc(signature=f8(f8), shorthand = '{0}*2')
def Double(a):
    return a * 2

@CREFunc(signature=f8(f8), shorthand = 'OnesDigit({0})')
def OnesDigit(a):
    return a % 10

@CREFunc(signature=f8(f8), shorthand = 'TensDigit({0})')
def TensDigit(a):
    return (a // 10) % 10


### Special Functions for Fractions 
###  --typically can be replaced with Multiply

@CREFunc(signature=f8(f8,f8,f8),
    shorthand='({0} / {1}) * {2}')
def ConvertNumerator(a, b, c):
    return (a / b) * c

@CREFunc(signature=f8(TextField, TextField),
    shorthand='Cross({0} * {1})')
def CrossMultiply(a, b):
    if('den' in a.id and 'den' in b.id):
        raise ValueError()
    if('num' in a.id and 'num' in b.id):
        raise ValueError()
    return (float(a.value) * float(b.value))

@CREFunc(signature=f8(TextField, TextField),
    shorthand='Across({0} * {1})')
def AcrossMultiply(a, b):
    if('den' in a.id and 'den' not in b.id):
        raise ValueError()
    if('num' in a.id and 'num' not in b.id):
        raise ValueError()

    return (float(a.value) * float(b.value))

# ----------------------
# Helpers
# ----------------------
deg = sp.pi/180

def _parse_vals(text):
    """Extract common symbols from the problem text into a dict of ints where present."""
    vals = {}
    for sym in ['a', 'b', 'c', 'A', 'B', 'C', 'scale', 'AB']:
        m = re.search(rf"\b{sym}\s*=\s*([0-9]+)", text)
        if m:
            vals[sym] = int(m.group(1))
    return vals

# @njit(cache=True)
def _any_match(pattern, text):
    """Safe matching helper that avoids regex on malformed planner strings."""
    try:
        t = str(text or "")
    except Exception:
        return False
    tl = t.lower()
    pl = str(pattern).lower()

    # Fast-path patterns used in this file.
    if pl == r"^\s*right\s*triangle":
        s = tl.strip()
        return s.startswith("right") and "triangle" in s
    if pl == r"^\s*asa\s*:\s*":
        return tl.strip().startswith("asa:")
    if pl == r"^\s*aas\s*:\s*":
        return tl.strip().startswith("aas:")
    if pl == r"\bssa\b":
        tokens = re.split(r"[^a-z0-9]+", tl)
        return "ssa" in tokens
    if pl == r"^\s*sas\s*:\s*":
        return tl.strip().startswith("sas:")
    if pl == r"^\s*sss\s*:\s*":
        return tl.strip().startswith("sss:")
    if pl == r"^\s*sim\s*:\s*":
        return tl.strip().startswith("sim:")
    if pl == r"^\s*area\s*:\s*":
        return tl.strip().startswith("area:")

    try:
        return re.search(pattern, t, flags=re.IGNORECASE) is not None
    except Exception:
        return False

# @njit(cache=True)
def _ok_any():
    # Accept anything non-empty; hint nudges what to put.
    return tuple([(re.compile(r".+"), "OK")])


# ----------------------
# Operator functions (return tuple of (regex, hint))
# ----------------------
@CREFunc(signature=string(string), shorthand = 'normalize_inputs({0})', nopython=False)
def normalize_inputs(init_value):
    # Accept any acknowledgement; this is a staging step.
    return "OK"

@CREFunc(signature=string(string), shorthand = 'classify_triangle({0})', nopython=False)
def classify_triangle(init_value):
    # Return a single canonical label to avoid regex-heavy matching in planner calls.
    txt = str(init_value or "").strip().lower()
    if txt.startswith("right") and "triangle" in txt:
        return "Right"
    if txt.startswith("asa:") or txt.startswith("aas:") or "ssa" in re.split(r"[^a-z0-9]+", txt):
        return "Sines"
    if txt.startswith("sas:") or txt.startswith("sss:"):
        return "Cosines"
    if txt.startswith("sim:"):
        return "Similarity"
    if txt.startswith("area:"):
        return "Area"
    # Fallback for under-specified prompts.
    return "Sines"

@CREFunc(signature=string(string), shorthand = 'apply_pythagorean({0})', nopython=False)
def apply_pythagorean(init_value):
    from sympy import latex
    vals = _parse_vals(init_value)
    a = vals.get('a')
    b = vals.get('b')
    if a is None or b is None:
        return ERROR_SENTINEL
    c = sp.sqrt(a*a + b*b)
    # Prefer simplified int if perfect square
    hint = latex(c)
    return str(hint)
    # return "OK"
    # ans = sstr(c, order="grlex")
    # return tuple([(re.compile(re.escape(ans)), hint)])

@CREFunc(signature=string(string), shorthand = 'use_trig_ratios({0})', nopython=False)
def use_trig_ratios(init_value):
    from sympy import latex
    # Prompt: Enter sin A for the right-triangle case
    vals = _parse_vals(init_value)
    a = vals.get('a')
    b = vals.get('b')
    if a is None or b is None:
        return ERROR_SENTINEL
    c = sp.Integer(a*a + b*b) ** sp.Rational(1, 2)
    sinA = a / c
    # hint = latex(sp.simplify(sinA))
    return sstr(sinA, order="grlex")
    # ans = sstr(sp.simplify(sinA), order="grlex")
    # return tuple([(re.compile(re.escape(ans)), hint)])


@CREFunc(signature=string(string), shorthand = 'resolve_ssa_ambiguity({0})',nopython=False)
def resolve_ssa_ambiguity(init_value):
    # If SSA present -> ambiguous; else unique
    tokens = re.split(r"[^a-z0-9]+", str(init_value or "").lower())
    ambiguous = "ssa" in tokens
    expected = "ambiguous" if ambiguous else "unique"
    return expected

    # Always accept the plain word first.
    # patterns = [(re.compile(expected, re.I), expected)]

    # # Also accept MathLive-style character-multiplication strings like
    # # u*(n*(i*(q*(e*u)))) for "unique", and analogous for "ambiguous".
    # if expected == "unique":
    #     # Allow both ...u.*e... and ...e.*u... at the end, since we observed both.
    #     patterns.append((re.compile(r"u.*n.*i.*q.*u.*e", re.I), expected))
    #     patterns.append((re.compile(r"u.*n.*i.*q.*e.*u", re.I), expected))
    # else:  # expected == "ambiguous"
    #     patterns.append((re.compile(r"a.*m.*b.*i.*g.*u.*o.*u.*s", re.I), expected))

    # return tuple(patterns)

@CREFunc(signature=string(string), shorthand = 'compute_missing_angles({0})', nopython=False)
def compute_missing_angles(init_value):
    # For ASA/AAS: compute B = 180 - (A + C)
    vals = _parse_vals(init_value)
    A = vals.get('A')
    C = vals.get('C')
    if A is None or C is None:
        return ERROR_SENTINEL
    B = 180 - (A + C)
    return sstr(B, order="grlex")
    # return tuple([(re.compile(str(B)), str(B))])

@CREFunc(signature=string(string), shorthand = 'compute_missing_sides({0})', nopython=False)
def compute_missing_sides(init_value):
    from sympy import latex
    # For ASA example in generator: a known, find b via Law of Sines
    vals = _parse_vals(init_value)
    A = vals.get('A')
    C = vals.get('C')
    a = vals.get('a')
    if A is None or C is None or a is None:
        return ERROR_SENTINEL
    B = 180 - (A + C)
    b = sp.nsimplify(a * sp.sin(B*deg) / sp.sin(A*deg))
    # hint = latex(b)
    # ans = sstr(b, order="grlex")
    return sstr(b, order="grlex")
    # return tuple([(re.compile(re.escape(ans)), hint)])

@CREFunc(signature=string(string), shorthand = 'compute_unknown_by_cosine({0})',nopython=False)
def compute_unknown_by_cosine(init_value):
    from sympy import latex
    # For SAS example in generator: compute c from a, b, C
    vals = _parse_vals(init_value)
    a = vals.get('a')
    b = vals.get('b')
    C = vals.get('C')
    if a is None or b is None or C is None:
        return ERROR_SENTINEL
    c2 = a*a + b*b - 2*a*b*sp.cos(C*deg)
    c = sp.sqrt(sp.simplify(c2))
    # hint = latex(c)
    # ans = sstr(c, order="grlex")
    # ans = re.compile(re.sub(r'([-+^()*])', r'\\\1', sstr(c, order="grlex")))
    return sstr(c, order="grlex")
    # return tuple([(ans, hint)])

# @CREFunc(signature=string(string), shorthand = 'backfill_with_sines_if_needed({0})', nopython=False)
# def backfill_with_sines_if_needed(init_value):
#     # Light-weight step: accept any non-empty.
#     return _ok_any()

@CREFunc(signature=string(string), shorthand = 'map_correspondence({0})', nopython=False)
def map_correspondence(init_value):
    # Accept any short token like A->A' or AB->A'B'
    return "Map corresponding vertices (e.g., A→A')"
    # return tuple([(re.compile(r".+"), "Map corresponding vertices (e.g., A→A')")])

@CREFunc(signature=string(string), shorthand = 'scale_sides_angles({0})', nopython=False)
def scale_sides_angles(init_value):
    from sympy import latex
    # For SIM example: scale=2, AB=5 -> A'B' = 10
    vals = _parse_vals(init_value)
    k = vals.get('scale')
    AB = vals.get('AB')
    if k is None or AB is None:
        return ERROR_SENTINEL
    scaled = sp.Integer(k*AB)
    # hint = latex(scaled)
    ans = sstr(scaled, order="grlex")
    return ans
    # return tuple([(re.compile(re.escape(ans)), hint)])

@CREFunc(signature=string(), shorthand = 'select_area_formula()', nopython=False)
def select_area_formula():
    from sympy import latex
    # Accept specific dropdown-friendly labels

    return "1/2·a·b·sin(C)"
    # return tuple([
    #     (re.compile(r"1/2·a·b·sin\(C\)", re.I), "1/2·a·b·sin(C)"),
    #     (re.compile(r"Heron", re.I), "Heron")
    # ])

@CREFunc(signature=string(string), shorthand = 'compute_area({0})', nopython=False)
def compute_area(init_value):
    from sympy import latex
    # Support 1/2·a·b·sin(C) or Heron's formula if all sides known
    vals = _parse_vals(init_value)
    a = vals.get('a')
    b = vals.get('b')
    C = vals.get('C')
    c = vals.get('c')

    area = None
    if a is not None and b is not None and C is not None:
        area = sp.nsimplify(sp.Rational(1, 2) * a * b * sp.sin(C*deg))
    elif a is not None and b is not None and c is not None:
        s = sp.Rational(a + b + c, 2)
        area = sp.nsimplify(sp.sqrt(s * (s - a) * (s - b) * (s - c)))
    if area is None:
        return ERROR_SENTINEL

    return sstr(area, order="grlex")
    # hint = latex(area)
    # ans = sstr(area, order="grlex")
    # return tuple([(re.compile(re.escape(ans)), hint)])

@CREFunc(signature=string(string), shorthand = 'consistency_checks({0})', nopython=False)
def consistency_checks(init_value):
    # Check common triangle consistency rules:
    # - Angle sum property
    # - Side length positivity
    # - Appropriate side/angle relationships
    vals = _parse_vals(init_value)
    A = vals.get('A')
    B = vals.get('B')
    C = vals.get('C')
    a = vals.get('a')
    b = vals.get('b')
    c = vals.get('c')

    hints = []
    if A is not None and B is not None and C is not None:
        # Angle sum property
        if A + B + C != 180:
            return "OK"  # Fail consistency
        hints.append("Angle sum property OK.")
    if a is not None and b is not None and c is not None:
        # Side length positivity
        if a <= 0 or b <= 0 or c <= 0:
            return "OK"  # Fail consistency
        hints.append("Side lengths positive.")
    if A is not None and a is not None and B is not None and b is not None:
        # Check relationships for given A, a, B, b
        if A > 90 and a <= b:
            return "OK"  # Fail consistency
        if B > 90 and b <= a:
            return "OK"  # Fail consistency
        hints.append("Angle-side relationships OK.")

    # If we have hints, return success with hints
    if hints:
        # return tuple([(re.compile("consistent"), "Consistent: " + ", ".join(hints))])
        return "Consistent: " + ", ".join(hints)

    # Default to OK
    return "OK"

# @CREFunc(signature=string(string), shorthand = 'report_solution({0})', nopython=False)
# def report_solution(init_value):
#     # Report the solution: list all sides, angles, and area if computable.
#     vals = _parse_vals(init_value)
#     A = vals.get('A')
#     B = vals.get('B')
#     C = vals.get('C')
#     a = vals.get('a')
#     b = vals.get('b')
#     c = vals.get('c')

#     # fill in missing angles
#     if a is not None and b is not None and c is not None:
#         if A is None:
#             A = sp.N(sp.acos((b**2 + c**2 - a**2) / (2*b*c)) / deg)
#         if B is None:
#             B = sp.N(sp.acos((a**2 + c**2 - b**2) / (2*a*c)) / deg)
#         if C is None:
#             C = sp.N(sp.acos((a**2 + b**2 - c**2) / (2*a*b)) / deg)
#     else:
#         if A is not None and B is not None and C is None:
#             C = 180 - (A + B)
#         elif A is not None and C is not None and B is None:
#             B = 180 - (A + C)
#         elif B is not None and C is not None and A is None:
#             A = 180 - (B + C)

#     area = None
#     if a is not None and b is not None and C is not None:
#         area = sp.nsimplify(sp.Rational(1, 2) * a * b * sp.sin(C*deg))
#     elif a is not None and b is not None and c is not None:
#         s = sp.Rational(a + b + c, 2)
#         area = sp.nsimplify(sp.sqrt(s * (s - a) * (s - b) * (s - c)))

#     # Canonical plain-text summary (used also for the hint)
#     solution = []
#     if A is not None:
#         solution.append(f"A = {sstr(sp.nsimplify(A))}")
#     if B is not None:
#         solution.append(f"B = {sstr(sp.nsimplify(B))}")
#     if C is not None:
#         solution.append(f"C = {sstr(sp.nsimplify(C))}")
#     if a is not None:
#         solution.append(f"a = {a}")
#     if b is not None:
#         solution.append(f"b = {b}")
#     if c is not None:
#         solution.append(f"c = {c}")
#     if area is not None:
#         solution.append(f"area = {sstr(area)}")

#     final_str = "Solution: " + ", ".join(solution)
#     patterns = [(re.compile(re.escape(final_str), re.I), final_str)]

#     # Lenient matcher that allows extra spacing or words between parts, keeping order
#     def part(label, value):
#         return rf"{label}\s*=\s*{re.escape(value)}"

#     parts = []
#     var_values = []
#     for lbl, val in [("A", A), ("B", B), ("C", C), ("a", a), ("b", b), ("c", c)]:
#         if val is not None:
#             v = sstr(sp.nsimplify(val)) if lbl.isupper() else str(val)
#             parts.append(part(lbl, v))
#             var_values.append((lbl, v))
#     if area is not None:
#         v = sstr(area)
#         parts.append(part("area", v))
#         var_values.append(("area", v))

#     if parts:
#         loose = r"Solution:\s*" + r".*".join(parts)
#         patterns.append((re.compile(loose, re.I), final_str))

#         # Accept MathLive/Sympy-style equation forms like Eq((S...)/X, value)
#         # We don't depend on exact letter-by-letter tokenization; match any LHS over X.
#         for var_name, var_val in var_values:
#             eq_pat = r"^Eq\(.*/" + re.escape(var_name) + r",\s*" + re.escape(var_val) + r"\)$"
#             patterns.append((re.compile(eq_pat, re.I), final_str))

#     return tuple(patterns)   # Return a string not a tuple of patterns


##### Define all CREFuncs above this line #####

register_all_funcs()


if __name__ == "__main__":
    # print(apply_pythagorean("HI"))
    print(apply_pythagorean("a=2 b=4"))

    print(use_trig_ratios("a=2 b=4"))

    print(resolve_ssa_ambiguity("SSA"))
    print(resolve_ssa_ambiguity("ASA"))


    print(compute_missing_angles("A=30 C=60"))
    print(compute_missing_sides("A=30 C=60 a=10"))
    print(compute_unknown_by_cosine("C=60 a=10 b=2"))

    print(map_correspondence(""))

    print(scale_sides_angles("scale=2, AB=5"))
    print(select_area_formula())
    # print(compute_area("C=60 a=10 b=2"))
    # normalize_inputs("FOO")    
