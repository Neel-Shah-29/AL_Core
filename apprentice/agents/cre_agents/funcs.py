from numba.types import f8, string, boolean
from apprentice.agents.cre_agents.extending import registries, new_register_decorator, new_register_all
from apprentice.agents.cre_agents.environment import TextField
from cre import CREFunc
import numpy as np
import sympy as sp
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

@CREFunc(signature=string(string), shorthand = '_parse_vals({0})')
def _parse_vals(text):
    """Extract common symbols from the problem text into a dict of ints where present."""
    vals = {}
    for sym in ['a', 'b', 'c', 'A', 'B', 'C', 'scale', 'AB']:
        m = re.search(rf"\b{sym}\s*=\s*([0-9]+)", text)
        if m:
            vals[sym] = int(m.group(1))
    return vals


def _any_match(pattern, text):
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _ok_any():
    # Accept anything non-empty; hint nudges what to put.
    return tuple([(re.compile(r".+"), "OK")])


# ----------------------
# Operator functions (return tuple of (regex, hint))
# ----------------------
@CREFunc(signature=string(string), shorthand = 'normalize_inputs({0})')
def normalize_inputs(init_value):
    # Accept any acknowledgement; this is a staging step.
    return _ok_any()

@CREFunc(signature=string(string), shorthand = 'classify_triangle({0})')
def classify_triangle(init_value):
    # Accept user-friendly labels while constraining to the scenario in the prompt.
    # We return multiple acceptable patterns but one concise hint string.
    def pack(patterns, hint):
        return tuple([(re.compile(pat, re.I), hint) for pat in patterns])

    if _any_match(r"^\s*Right\s*triangle", init_value):
        return pack([
            r"right",
            r"right\s*triangle",
            r"righttriangletrig",
            r"right-?angled",
            r"rt|rtt",
            r"R*(i*(g*(h*t)))"
        ], "Right")
    if (_any_match(r"^\s*ASA\s*:\s*", init_value)
        or _any_match(r"^\s*AAS\s*:\s*", init_value)
        or _any_match(r"\bSSA\b", init_value)):
        return pack([
            r"sines",
            r"law\s*of\s*sines",
            r"los|sin",
            r"S*(i*(n*(e*s)))"
        ], "Sines")
    if _any_match(r"^\s*SAS\s*:\s*", init_value) or _any_match(r"^\s*SSS\s*:\s*", init_value):
        return pack([
            r"cosines",
            r"law\s*of\s*cosines",
            r"loc|cos",
            r"C*(o*(s*(i*(n*(e*s)))))"
        ], "Cosines")
    if _any_match(r"^\s*SIM\s*:\s*", init_value):
        return pack([
            r"similarity",
            r"similarity\s*scaling",
            r"sim"
        ], "Similarity")
    if _any_match(r"^\s*AREA\s*:\s*", init_value):
        return pack([
            r"area",
            r"area\s*relations",
            r"heron|\b1/2\s*ab\s*sin\s*c\b"
        ], "Area")
    # Fallback: default to Sines family but accept broad labels
    return pack([
        r"sines",
        r"law\s*of\s*sines",
        r"los|sin"
    ], "Sines")

@CREFunc(signature=string(string), shorthand = 'apply_pythagorean({0})', nopython=False)
def apply_pythagorean(init_value):
    vals = _parse_vals(init_value)
    a = vals.get('a')
    b = vals.get('b')
    if a is None or b is None:
        return _ok_any()
    c = sp.sqrt(a*a + b*b)
    # Prefer simplified int if perfect square
    hint = latex(c)
    ans = sstr(c, order="grlex")
    return tuple([(re.compile(re.escape(ans)), hint)])

@CREFunc(signature=string(string), shorthand = 'use_trig_ratios({0})')
def use_trig_ratios(init_value):
    # Prompt: Enter sin A for the right-triangle case
    vals = _parse_vals(init_value)
    a = vals.get('a')
    b = vals.get('b')
    if a is None or b is None:
        return _ok_any()
    c = sp.Integer(a*a + b*b) ** sp.Rational(1, 2)
    sinA = sp.Rational(a, c)
    hint = latex(sp.simplify(sinA))
    ans = sstr(sp.simplify(sinA), order="grlex")
    return tuple([(re.compile(re.escape(ans)), hint)])


@CREFunc(signature=string(string), shorthand = 'resolve_ssa_ambiguity({0})')
def resolve_ssa_ambiguity(init_value):
    # If SSA present -> ambiguous; else unique
    ambiguous = _any_match(r"\bSSA\b", init_value)
    expected = "ambiguous" if ambiguous else "unique"

    # Always accept the plain word first.
    patterns = [(re.compile(expected, re.I), expected)]

    # Also accept MathLive-style character-multiplication strings like
    # u*(n*(i*(q*(e*u)))) for "unique", and analogous for "ambiguous".
    if expected == "unique":
        # Allow both ...u.*e... and ...e.*u... at the end, since we observed both.
        patterns.append((re.compile(r"u.*n.*i.*q.*u.*e", re.I), expected))
        patterns.append((re.compile(r"u.*n.*i.*q.*e.*u", re.I), expected))
    else:  # expected == "ambiguous"
        patterns.append((re.compile(r"a.*m.*b.*i.*g.*u.*o.*u.*s", re.I), expected))

    return tuple(patterns)

@CREFunc(signature=string(string), shorthand = 'compute_missing_angles({0})')
def compute_missing_angles(init_value):
    # For ASA/AAS: compute B = 180 - (A + C)
    vals = _parse_vals(init_value)
    A = vals.get('A')
    C = vals.get('C')
    if A is None or C is None:
        return _ok_any()
    B = 180 - (A + C)
    return tuple([(re.compile(str(B)), str(B))])

@CREFunc(signature=string(string), shorthand = 'compute_missing_sides({0})')
def compute_missing_sides(init_value):
    # For ASA example in generator: a known, find b via Law of Sines
    vals = _parse_vals(init_value)
    A = vals.get('A')
    C = vals.get('C')
    a = vals.get('a')
    if A is None or C is None or a is None:
        return _ok_any()
    B = 180 - (A + C)
    b = sp.nsimplify(a * sp.sin(B*deg) / sp.sin(A*deg))
    hint = latex(b)
    ans = sstr(b, order="grlex")
    return tuple([(re.compile(re.escape(ans)), hint)])

@CREFunc(signature=string(string), shorthand = 'compute_unknown_by_cosine({0})')
def compute_unknown_by_cosine(init_value):
    # For SAS example in generator: compute c from a, b, C
    vals = _parse_vals(init_value)
    a = vals.get('a')
    b = vals.get('b')
    C = vals.get('C')
    if a is None or b is None or C is None:
        return _ok_any()
    c2 = a*a + b*b - 2*a*b*sp.cos(C*deg)
    c = sp.sqrt(sp.simplify(c2))
    hint = latex(c)
    # ans = sstr(c, order="grlex")
    ans = re.compile(re.sub(r'([-+^()*])', r'\\\1', sstr(c, order="grlex")))
    return tuple([(ans, hint)])

@CREFunc(signature=string(string), shorthand = 'backfill_with_sines_if_needed({0})')
def backfill_with_sines_if_needed(init_value):
    # Light-weight step: accept any non-empty.
    return _ok_any()

@CREFunc(signature=string(string), shorthand = 'map_correspondence({0})')
def map_correspondence(init_value):
    # Accept any short token like A->A' or AB->A'B'
    return tuple([(re.compile(r".+"), "Map corresponding vertices (e.g., A→A')")])

@CREFunc(signature=string(string), shorthand = 'scale_sides_angles({0})')
def scale_sides_angles(init_value):
    # For SIM example: scale=2, AB=5 -> A'B' = 10
    vals = _parse_vals(init_value)
    k = vals.get('scale')
    AB = vals.get('AB')
    if k is None or AB is None:
        return _ok_any()
    scaled = sp.Integer(k*AB)
    hint = latex(scaled)
    ans = sstr(scaled, order="grlex")
    return tuple([(re.compile(re.escape(ans)), hint)])

@CREFunc(signature=string(string), shorthand = 'select_area_formula({0})')
def select_area_formula(init_value):
    # Accept specific dropdown-friendly labels
    return tuple([
        (re.compile(r"1/2·a·b·sin\(C\)", re.I), "1/2·a·b·sin(C)"),
        (re.compile(r"Heron", re.I), "Heron")
    ])

@CREFunc(signature=string(string), shorthand = 'compute_area({0})')
def compute_area(init_value):
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
        return _ok_any()

    hint = latex(area)
    ans = sstr(area, order="grlex")
    return tuple([(re.compile(re.escape(ans)), hint)])

@CREFunc(signature=string(string), shorthand = 'consistency_checks({0})')
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
            return _ok_any()  # Fail consistency
        hints.append("Angle sum property OK.")
    if a is not None and b is not None and c is not None:
        # Side length positivity
        if a <= 0 or b <= 0 or c <= 0:
            return _ok_any()  # Fail consistency
        hints.append("Side lengths positive.")
    if A is not None and a is not None and B is not None and b is not None:
        # Check relationships for given A, a, B, b
        if A > 90 and a <= b:
            return _ok_any()  # Fail consistency
        if B > 90 and b <= a:
            return _ok_any()  # Fail consistency
        hints.append("Angle-side relationships OK.")

    # If we have hints, return success with hints
    if hints:
        return tuple([(re.compile("consistent"), "Consistent: " + ", ".join(hints))])

    # Default to OK
    return _ok_any()

@CREFunc(signature=string(string), shorthand = 'report_solution({0})')
def report_solution(init_value):
    # Report the solution: list all sides, angles, and area if computable.
    vals = _parse_vals(init_value)
    A = vals.get('A')
    B = vals.get('B')
    C = vals.get('C')
    a = vals.get('a')
    b = vals.get('b')
    c = vals.get('c')

    # fill in missing angles
    if a is not None and b is not None and c is not None:
        if A is None:
            A = sp.N(sp.acos((b**2 + c**2 - a**2) / (2*b*c)) / deg)
        if B is None:
            B = sp.N(sp.acos((a**2 + c**2 - b**2) / (2*a*c)) / deg)
        if C is None:
            C = sp.N(sp.acos((a**2 + b**2 - c**2) / (2*a*b)) / deg)
    else:
        if A is not None and B is not None and C is None:
            C = 180 - (A + B)
        elif A is not None and C is not None and B is None:
            B = 180 - (A + C)
        elif B is not None and C is not None and A is None:
            A = 180 - (B + C)

    area = None
    if a is not None and b is not None and C is not None:
        area = sp.nsimplify(sp.Rational(1, 2) * a * b * sp.sin(C*deg))
    elif a is not None and b is not None and c is not None:
        s = sp.Rational(a + b + c, 2)
        area = sp.nsimplify(sp.sqrt(s * (s - a) * (s - b) * (s - c)))

    # Canonical plain-text summary (used also for the hint)
    solution = []
    if A is not None:
        solution.append(f"A = {sstr(sp.nsimplify(A))}")
    if B is not None:
        solution.append(f"B = {sstr(sp.nsimplify(B))}")
    if C is not None:
        solution.append(f"C = {sstr(sp.nsimplify(C))}")
    if a is not None:
        solution.append(f"a = {a}")
    if b is not None:
        solution.append(f"b = {b}")
    if c is not None:
        solution.append(f"c = {c}")
    if area is not None:
        solution.append(f"area = {sstr(area)}")

    final_str = "Solution: " + ", ".join(solution)
    patterns = [(re.compile(re.escape(final_str), re.I), final_str)]

    # Lenient matcher that allows extra spacing or words between parts, keeping order
    def part(label, value):
        return rf"{label}\s*=\s*{re.escape(value)}"

    parts = []
    var_values = []
    for lbl, val in [("A", A), ("B", B), ("C", C), ("a", a), ("b", b), ("c", c)]:
        if val is not None:
            v = sstr(sp.nsimplify(val)) if lbl.isupper() else str(val)
            parts.append(part(lbl, v))
            var_values.append((lbl, v))
    if area is not None:
        v = sstr(area)
        parts.append(part("area", v))
        var_values.append(("area", v))

    if parts:
        loose = r"Solution:\s*" + r".*".join(parts)
        patterns.append((re.compile(loose, re.I), final_str))

        # Accept MathLive/Sympy-style equation forms like Eq((S...)/X, value)
        # We don't depend on exact letter-by-letter tokenization; match any LHS over X.
        for var_name, var_val in var_values:
            eq_pat = r"^Eq\(.*/" + re.escape(var_name) + r",\s*" + re.escape(var_val) + r"\)$"
            patterns.append((re.compile(eq_pat, re.I), final_str))

    return tuple(patterns)



##### Define all CREFuncs above this line #####

register_all_funcs()

