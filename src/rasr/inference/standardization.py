from __future__ import annotations

import re

import numpy as np

_EPS = 1e-12

_MULT_OPS: dict[str, float] = {
    "mult2": 2.0, "mult3": 3.0, "mult4": 4.0, "mult5": 5.0,
}
_DIV_OPS: dict[str, float] = {
    "div2": 2.0, "div3": 3.0, "div4": 4.0, "div5": 5.0,
}


def _is_const(tok: str) -> bool:
    if tok == "<constant>":
        return True
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _fmt(val: float) -> str:
    return f"{val:.8g}"


def _is_var(tok: str) -> bool:
    return bool(re.match(r"^x\d+$", tok))


def _apply_affine(toks: list[str], scale: float, offset: float) -> list[str]:
    """Return tokens for ``scale * expr + offset``, folding into existing constants where possible."""
    if abs(scale - 1.0) < _EPS and abs(offset) < _EPS:
        return toks

    if len(toks) == 1 and _is_const(toks[0]):
        return [_fmt(float(toks[0]) * scale + offset)]

    result = list(toks)

    # +(left, <const>) is the common model output form, so distribute through it.
    if result[0] == "+" and len(result) > 2 and _is_const(result[-1]):
        right_val = float(result[-1])
        left_toks = result[1:-1]
        if abs(scale - 1.0) > _EPS:
            left_toks = _apply_affine(left_toks, scale, 0.0)
            right_val *= scale
        if abs(offset) > _EPS:
            right_val += offset
        return ["+"] + left_toks + [_fmt(right_val)]

    if abs(scale - 1.0) > _EPS:
        if result[0] == "*" and len(result) > 1 and _is_const(result[1]):
            result = ["*", _fmt(float(result[1]) * scale)] + result[2:]
        else:
            result = ["*", _fmt(scale)] + result

    if abs(offset) > _EPS:
        result = ["+"] + result + [_fmt(offset)]

    return result


def _remap(
    tokens: list[str],
    pos: int,
    fitted: list[float],
    ci: list[int],
    X_mean: np.ndarray,
    X_scale: np.ndarray,
    op_arity: dict[str, int],
) -> tuple[list[str], float, float, int]:
    """Remap one subtree starting at *pos*.

    Returns ``(new_tokens, scale, offset, end_pos)`` where
    ``f_norm(x_norm) = scale * new_expr(x_orig) + offset``.
    """
    tok = tokens[pos]
    if _is_var(tok):
        i = int(tok[1:]) - 1
        return [tok], 1.0 / X_scale[i], -X_mean[i] / X_scale[i], pos + 1
    if _is_const(tok):
        val = fitted[ci[0]] if ci[0] < len(fitted) else (0.0 if tok == "<constant>" else float(tok))
        ci[0] += 1
        return [_fmt(val)], 1.0, 0.0, pos + 1
    if tok == "+":
        lt, sl, ol, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        rt, sr, or_, pos = _remap(tokens, pos, fitted, ci, X_mean, X_scale, op_arity)
        if abs(sl - 1.0) > _EPS:
            lt = ["*", _fmt(sl)] + lt
        if abs(sr - 1.0) > _EPS:
            rt = ["*", _fmt(sr)] + rt
        total = ol + or_
        if abs(total) > _EPS:
            if len(rt) == 1 and _is_const(rt[0]):
                rt = [_fmt(float(rt[0]) + total)]
                total = 0.0
            elif len(lt) == 1 and _is_const(lt[0]):
                lt = [_fmt(float(lt[0]) + total)]
                total = 0.0
        return ["+"] + lt + rt, 1.0, total, pos
    if tok == "-":
        lt, sl, ol, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        rt, sr, or_, pos = _remap(tokens, pos, fitted, ci, X_mean, X_scale, op_arity)
        if abs(sl - 1.0) > _EPS:
            lt = ["*", _fmt(sl)] + lt
        if abs(sr - 1.0) > _EPS:
            rt = ["*", _fmt(sr)] + rt
        total = ol - or_
        if abs(total) > _EPS:
            if len(lt) == 1 and _is_const(lt[0]):
                lt = [_fmt(float(lt[0]) + total)]
                total = 0.0
            elif len(rt) == 1 and _is_const(rt[0]):
                rt = [_fmt(float(rt[0]) - total)]
                total = 0.0
        return ["-"] + lt + rt, 1.0, total, pos
    if tok == "*":
        lt, sl, ol, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        rt, sr, or_, pos = _remap(tokens, pos, fitted, ci, X_mean, X_scale, op_arity)
        if len(lt) == 1 and _is_const(lt[0]):
            # C·(sr·r + or_) = (C·sr)·r + C·or_.
            C = float(lt[0])
            return ["*", _fmt(C * sr)] + rt, 1.0, C * or_, pos
        if len(rt) == 1 and _is_const(rt[0]):
            # (sl·l + ol)·C = (sl·C)·l + ol·C.
            C = float(rt[0])
            return ["*"] + lt + [_fmt(sl * C)], 1.0, ol * C, pos
        return ["*"] + _apply_affine(lt, sl, ol) + _apply_affine(rt, sr, or_), 1.0, 0.0, pos
    if tok == "/":
        lt, sl, ol, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        rt, sr, or_, pos = _remap(tokens, pos, fitted, ci, X_mean, X_scale, op_arity)
        if len(rt) == 1 and _is_const(rt[0]) and abs(or_) < _EPS:
            C = sr * float(rt[0])
            denom_str = _fmt(C)
            if abs(C) > _EPS:
                return ["/"] + lt + [denom_str], sl, ol / C, pos
        if len(lt) == 1 and _is_const(lt[0]) and abs(ol) < _EPS:
            return ["/", lt[0]] + _apply_affine(rt, sr, or_), 1.0, 0.0, pos
        return ["/"] + _apply_affine(lt, sl, ol) + _apply_affine(rt, sr, or_), 1.0, 0.0, pos
    if tok == "neg":
        ct, sc, oc, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        # neg(sc·c + oc) = sc·neg(c) − oc.
        return ["neg"] + ct, sc, -oc, pos
    if tok == "pow":
        bt, sb, ob, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        et, se, oe, pos = _remap(tokens, pos, fitted, ci, X_mean, X_scale, op_arity)
        return ["pow"] + _apply_affine(bt, sb, ob) + _apply_affine(et, se, oe), 1.0, 0.0, pos
    # mult_n(sc·c + oc) = (n·sc)·c + n·oc.
    if tok in _MULT_OPS:
        factor = _MULT_OPS[tok]
        ct, sc, oc, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        return ["*", _fmt(factor * sc)] + ct, 1.0, factor * oc, pos
    if tok in _DIV_OPS:
        denom = _DIV_OPS[tok]
        ct, sc, oc, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        return ["*", _fmt(sc / denom)] + ct, 1.0, oc / denom, pos
    # Nonlinear ops can't propagate the affine map, so absorb it into each argument.
    arity = int(op_arity.get(tok, 0))
    new_toks = [tok]
    for _ in range(arity):
        ct, sc, oc, pos = _remap(tokens, pos + 1, fitted, ci, X_mean, X_scale, op_arity)
        new_toks.extend(_apply_affine(ct, sc, oc))
    if arity == 0:
        pos += 1
    return new_toks, 1.0, 0.0, pos


def remap_to_original_space(
    tokens: list[str],
    fitted_constants: np.ndarray,
    X_mean: np.ndarray,
    X_scale: np.ndarray,
    y_mean: float,
    y_scale: float,
    op_arity: dict[str, int],
    variables: set[str],
    precision: int = 8,
) -> tuple[list[str], np.ndarray]:
    """Remap expression constants from normalized to original space; ``variables`` is unused."""
    ci = [0]
    fitted = list(fitted_constants)

    new_toks, scale, offset, _ = _remap(tokens, 0, fitted, ci, X_mean, X_scale, op_arity)

    # y_orig = y_scale·(scale·expr + offset) + y_mean.
    final_scale = scale * y_scale
    final_offset = offset * y_scale + y_mean
    new_toks = _apply_affine(new_toks, final_scale, final_offset)

    if precision != 8:
        out: list[str] = []
        for t in new_toks:
            if _is_const(t) and t != "<constant>":
                try:
                    out.append(f"{float(t):.{precision}g}")
                except ValueError:
                    out.append(t)
            else:
                out.append(t)
        new_toks = out

    consts = [float(t) for t in new_toks if _is_const(t) and t != "<constant>"]
    return new_toks, np.array(consts, dtype=float)
