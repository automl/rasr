from __future__ import annotations

import math


def _format_constant_sci(value: float, sig_figs: int) -> list[str]:
    """Format a float in scientific notation as per-digit tokens, with the exponent clamped to one digit."""
    if not math.isfinite(value):
        raise ValueError(f"Cannot represent non-finite value {value!r} in scientific notation.")

    if value < 0:
        prefix = ["neg"]
        abs_val = -value
    else:
        prefix = []
        abs_val = value

    if abs_val == 0.0:
        return prefix + ["0", "e+", "0"]

    exp = int(math.floor(math.log10(abs_val)))
    mantissa = abs_val / (10.0 ** exp)

    decimal_places = max(0, sig_figs - 1)
    mantissa_rounded = round(mantissa, decimal_places)

    # Rounding 9.999... can push mantissa to 10.
    if mantissa_rounded >= 10.0:
        mantissa_rounded /= 10.0
        exp += 1

    if decimal_places == 0:
        mantissa_digits = str(int(round(mantissa_rounded)))
    else:
        mantissa_digits = f"{mantissa_rounded:.{decimal_places}f}".rstrip("0").replace(".", "")

    exp_sign = "e+" if exp >= 0 else "e-"
    exp_digit = str(min(abs(exp), 9))

    return prefix + list(mantissa_digits) + [exp_sign, exp_digit]


class ExpressionWrapper:
    """Wraps a prefix expression to describe standardized y = (f(X) - μ) / σ."""

    def __init__(self, operator_arity: dict[str, int]) -> None:
        self.operator_arity = operator_arity

    @staticmethod
    def _float_tokens(value: float) -> tuple[str, ...]:
        """Format a float constant as tokens at full precision."""
        if value < 0:
            return ("neg", str(abs(value)))
        return (str(value),)

    @staticmethod
    def round_constants(
        tokens: tuple[str, ...],
        sig_figs: int,
        strategy: str = "mantissa_exponent",
        constant_token: str = "<constant>",
    ) -> tuple[str, ...]:
        """Convert every numeric constant to its output-token representation."""
        result: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "neg" and i + 1 < len(tokens):
                try:
                    val = -float(tokens[i + 1])
                    if math.isfinite(val):
                        if strategy == "placeholder":
                            result.append(constant_token)
                        else:
                            result.extend(_format_constant_sci(val, sig_figs))
                        i += 2
                        continue
                except ValueError:
                    pass
            try:
                val = float(token)
                if math.isfinite(val):
                    if strategy == "placeholder":
                        result.append(constant_token)
                    else:
                        result.extend(_format_constant_sci(val, sig_figs))
                    i += 1
                    continue
            except ValueError:
                pass
            result.append(token)
            i += 1
        return tuple(result)

    def _subtree_end(self, tokens: tuple[str, ...], pos: int) -> int:
        """Return the index one past the subtree rooted at pos."""
        token = tokens[pos]
        if token == "neg":
            return pos + 2
        try:
            float(token)
            return pos + 1
        except ValueError:
            pass
        arity = self.operator_arity.get(token, 0)
        end = pos + 1
        for _ in range(arity):
            end = self._subtree_end(tokens, end)
        return end

    def _extract_additive_const(
        self, tokens: tuple[str, ...], pos: int
    ) -> tuple[int, float, tuple[str, ...] | None, bool]:
        """Split top-level additive literal constants off the subtree at pos.

        Returns ``(end_pos, const_sum, remaining, found_any)``; remaining is None if the subtree is entirely constant.
        """
        token = tokens[pos]

        if token == "neg":
            try:
                val = -float(tokens[pos + 1])
                return pos + 2, val, None, True
            except (ValueError, IndexError):
                pass

        try:
            val = float(token)
            return pos + 1, val, None, True
        except ValueError:
            pass

        if token == "+" and self.operator_arity.get(token) == 2:
            left_end, left_const, left_rem, left_found = self._extract_additive_const(tokens, pos + 1)
            right_end, right_const, right_rem, right_found = self._extract_additive_const(tokens, left_end)
            total = left_const + right_const
            found = left_found or right_found

            if left_rem is None and right_rem is None:
                return right_end, total, None, found
            elif left_rem is None:
                return right_end, total, right_rem, found
            elif right_rem is None:
                return right_end, total, left_rem, found
            else:
                return right_end, total, ("+",) + left_rem + right_rem, found

        end = self._subtree_end(tokens, pos)
        return end, 0.0, tokens[pos:end], False

    def wrap(
        self,
        expression: tuple[str, ...],
        mu_y: float,
        sigma_y: float,
    ) -> tuple[str, ...]:
        """Rewrite *expression* to describe standardized y = (f(X) - μ_y) / σ_y.

        μ_y is folded into existing top-level additive constants; constants are left unrounded.
        """
        if mu_y == 0.0 and sigma_y == 1.0:
            return expression
        tok = self._float_tokens

        if mu_y != 0.0:
            _, additive_const, remaining, found = self._extract_additive_const(expression, 0)
            if found:
                new_const = additive_const - mu_y
                result = tok(new_const) if remaining is None else ("+",) + tok(new_const) + remaining
                return result if sigma_y == 1.0 else ("/",) + result + tok(sigma_y)

        if mu_y == 0.0:
            return ("/",) + expression + tok(sigma_y)
        return ("/", "-") + expression + tok(mu_y) + tok(sigma_y)
