"""Data-informed constant sampling for expression skeletons."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import truncnorm as _truncnorm


class BottomUpConstantSampler:
    """Sample constants for an expression skeleton using empirical X statistics."""

    def __init__(self, config: dict) -> None:
        self.cfg = config

    def _sample_op_constants(
        self, op: str, raw_vals: np.ndarray, *, scale_only: bool = False, shift_only: bool = False
    ) -> tuple[float, float]:
        """Sample (c0, c1) so that ``c0 * raw_vals + c1`` lies in *op*'s preferred input regime."""
        cfg = self.cfg[op]
        # TODO: The data statistics currently cancel out for c0 and c1; consider letting them influence sampling.
        positive_domain = op in ("log", "pow1_2", "pow1_4")

        if shift_only:
            c0 = 1.0
        else:
            sigma_data = max(float(np.std(raw_vals)), 1e-8)
            c0 = float(np.exp(np.random.normal(cfg["c0_log_mean"] - np.log(sigma_data), cfg["c0_log_std"])))
            if not positive_domain and np.random.rand() < 0.5:
                c0 = -c0

        if scale_only:
            return float(c0), 0.0

        if positive_domain:
            d = float(np.exp(np.random.normal(np.log(cfg["c1_center"]), cfg["c1_std"])))
            c1 = float(d - np.min(c0 * raw_vals))
        elif op == "inv":
            c1 = self._sample_denominator_shift(c0 * raw_vals)
        elif op == "abs":
            scaled = c0 * raw_vals
            c1_mean = float(cfg["c1_mean"]) - float(np.mean(scaled))
            c1 = self._sample_straddle_shift(scaled, c1_mean, cfg["c1_std"], cfg["c1_straddle_margin"])
        else:
            mu_data = float(np.mean(raw_vals))
            c1 = float(np.random.normal(cfg["c1_mean"] - c0 * mu_data, cfg["c1_std"]))

        return float(c0), float(c1)

    def _sample_straddle_shift(self, vals: np.ndarray, c1_mean: float, c1_std: float, margin: float = 0.1) -> float:
        """Sample a shift c so that (vals + c) straddles zero by at least ``margin * range(vals)``."""
        lo = float(-np.max(vals))
        hi = float(-np.min(vals))
        if lo >= hi:
            return float(-np.mean(vals))
        delta = margin * (hi - lo)
        lo_eff = lo + delta
        hi_eff = hi - delta
        if lo_eff >= hi_eff:
            lo_eff, hi_eff = (lo + hi) / 2.0 - 1e-8, (lo + hi) / 2.0 + 1e-8
        a = (lo_eff - c1_mean) / c1_std
        b = (hi_eff - c1_mean) / c1_std
        return float(_truncnorm.rvs(a, b, loc=c1_mean, scale=c1_std))

    def _sample_denominator_shift(self, raw_vals: np.ndarray, *, allow_straddle: bool = True) -> float:
        """Sample a shift so that (raw_vals + shift) stays away from zero.

        ``allow_straddle=False`` forces a one-sided domain, for when the constant itself is the denominator.
        """
        cfg = self.cfg["inv"]
        probs = np.array(cfg["domain_probs"], dtype=float)
        if not allow_straddle:
            probs[2] = 0.0
        probs /= probs.sum()
        cat = int(np.random.choice(len(probs), p=probs))

        if cat == 2:
            return float(np.random.normal(-float(np.mean(raw_vals)), cfg["c1_std"]))

        log_d = float(np.random.normal(cfg["c0_log_mean"], cfg["c0_log_std"]))
        d = float(np.exp(log_d))
        if cat == 1:
            return float(d - np.min(raw_vals))
        else:
            return float(-d - np.max(raw_vals))

    def _parse_prefix(
        self,
        tokens: tuple | list,
        variable_set: set[str],
        operator_arity: dict[str, int],
    ) -> dict[str, Any]:
        """Parse a prefix-notation token sequence into a nested dict tree."""
        tokens = list(tokens)
        idx = [0]
        const_counter = [0]

        def parse() -> dict[str, Any]:
            if idx[0] >= len(tokens):
                ci = const_counter[0]
                const_counter[0] += 1
                return {"type": "const", "const_idx": ci}

            token = tokens[idx[0]]
            idx[0] += 1
            arity = operator_arity.get(token, 0)

            if arity == 0:
                if token == "<constant>":
                    ci = const_counter[0]
                    const_counter[0] += 1
                    return {"type": "const", "const_idx": ci}
                if token in variable_set:
                    return {"type": "var", "name": token}
                try:
                    val = float(token)
                    return {"type": "number", "value": val}
                except ValueError:
                    return {"type": "var", "name": token}

            children = [parse() for _ in range(arity)]
            return {"type": "op", "name": token, "arity": arity, "children": children}

        return parse()

    def _apply_op(
        self, op: str, left: np.ndarray, right: np.ndarray | None = None
    ) -> np.ndarray:
        with np.errstate(all="ignore"):
            if right is None:
                if op == "sin":
                    return np.sin(left)
                if op == "cos":
                    return np.cos(left)
                if op == "exp":
                    return np.exp(np.clip(left, -80.0, 80.0))
                if op == "log":
                    return np.log(np.where(left > 1e-10, left, 1e-10))
                if op == "tanh":
                    return np.tanh(left)
                if op == "neg":
                    return -left
                if op == "abs":
                    return np.abs(left)
                if op == "inv":
                    safe = np.where(np.abs(left) > 1e-8, left, np.sign(left + 1e-10) * 1e-8)
                    return 1.0 / safe
                if op == "pow2":
                    return left * left
                if op == "pow3":
                    return left ** 3
                if op == "pow4":
                    return left ** 4
                if op == "pow5":
                    return left ** 5
                if op == "pow1_2":
                    return np.sqrt(np.maximum(left, 0.0))
                if op == "pow1_3":
                    return np.cbrt(left)
                if op == "pow1_4":
                    return np.power(np.maximum(left, 0.0), 0.25)
                if op == "pow1_5":
                    return np.sign(left) * np.power(np.abs(left), 0.2)
                return left
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                safe = np.where(np.abs(right) > 1e-8, right, np.sign(right + 1e-10) * 1e-8)
                return left / safe
        return left

    @staticmethod
    def _is_scale_wrapper(node: dict) -> bool:
        """True if node is (* <constant> subtree) in either child order."""
        return (
            node["type"] == "op" and node["name"] == "*" and node["arity"] == 2
            and (node["children"][0]["type"] == "const" or node["children"][1]["type"] == "const")
        )

    @staticmethod
    def _is_shift_wrapper(node: dict) -> bool:
        """True if node is (+ <constant> subtree) or (+ subtree <constant>)."""
        return (
            node["type"] == "op" and node["name"] == "+" and node["arity"] == 2
            and (node["children"][0]["type"] == "const") != (node["children"][1]["type"] == "const")
        )

    @classmethod
    def _is_summand_pair(cls, node: dict) -> bool:
        """True if node is (+/- (* <c0> A) (* <c1> B)) — two scaled summands, no bias."""
        return (
            node["type"] == "op" and node["name"] in ("+", "-") and node["arity"] == 2
            and cls._is_scale_wrapper(node["children"][0])
            and cls._is_scale_wrapper(node["children"][1])
        )

    @classmethod
    def _is_summand_pair_with_bias(cls, node: dict) -> bool:
        """True if node is (+ summand_pair <const>) or (+ <const> summand_pair)."""
        if not (node["type"] == "op" and node["name"] == "+" and node["arity"] == 2):
            return False
        c0, c1 = node["children"]
        return (
            (cls._is_summand_pair(c0) and c1["type"] == "const")
            or (cls._is_summand_pair(c1) and c0["type"] == "const")
        )

    @staticmethod
    def _scale_parts(node: dict) -> tuple[dict, dict]:
        """Extract (const_node, subtree_node) from a scale wrapper, order-agnostic."""
        if node["children"][0]["type"] == "const":
            return node["children"][0], node["children"][1]
        return node["children"][1], node["children"][0]

    @classmethod
    def _is_affine_wrapper(cls, node: dict) -> bool:
        """True if node is (+ (* <c0> subtree) <c1>) in either child order."""
        if not (node["type"] == "op" and node["name"] == "+" and node["arity"] == 2):
            return False
        c0, c1 = node["children"]
        return (
            (cls._is_scale_wrapper(c0) and c1["type"] == "const")
            or (cls._is_scale_wrapper(c1) and c0["type"] == "const")
        )

    def _set_summand_scales_for_op(
        self, op: str, a_vals: np.ndarray, b_vals: np.ndarray
    ) -> tuple[float, float]:
        """Like _set_summand_scales, but the combined sigma targets *op*'s preferred input scale."""
        cfg = self.cfg[op]
        sigma_a = max(float(np.std(a_vals)), 1e-8)
        sigma_b = max(float(np.std(b_vals)), 1e-8)
        sigma_combined = max(float(np.std(np.concatenate([a_vals, b_vals]))), 1e-8)
        target_sigma = sigma_combined * float(np.exp(
            np.random.normal(cfg["c0_log_mean"], cfg["c0_log_std"])
        ))
        log_ratio = float(np.random.normal(0.0, 0.7))
        c0 = float(target_sigma * np.exp(log_ratio / 2.0) / sigma_a)
        c1 = float(target_sigma * np.exp(-log_ratio / 2.0) / sigma_b)
        if np.random.rand() < 0.5:
            c0 = -c0
        if np.random.rand() < 0.5:
            c1 = -c1
        return c0, c1

    @classmethod
    def _flatten_sum_chain(cls, node: dict, sign: int = 1) -> tuple[list, list]:
        """Split a +/- chain into signed scale-wrapper terms and signed other terms."""
        if node["type"] == "op" and node["name"] in ("+", "-"):
            right_sign = sign if node["name"] == "+" else -sign
            sl, ol = cls._flatten_sum_chain(node["children"][0], sign)
            sr, or_ = cls._flatten_sum_chain(node["children"][1], right_sign)
            return sl + sr, ol + or_
        elif cls._is_scale_wrapper(node):
            c_node, sub_node = cls._scale_parts(node)
            return [(c_node, sub_node, sign)], []
        else:
            return [], [(node, sign)]

    def _fill_sum_chain_jointly(
        self,
        node: dict,
        x_data: np.ndarray,
        variables: list[str],
        constants: list,
        operator_arity: dict[str, int],
    ) -> np.ndarray:
        """Jointly set all outer scale constants in a +/- chain, anchored to the geometric mean of the raw stds."""
        n = len(x_data)
        scale_terms, others = self._flatten_sum_chain(node)

        raw_vals_list = [
            self._fill_and_evaluate(sub, x_data, variables, constants, operator_arity)
            for _, sub, _ in scale_terms
        ]

        others_val = np.zeros(n, dtype=np.float64)
        for other_node, other_sign in others:
            others_val += other_sign * self._fill_and_evaluate(other_node, x_data, variables, constants, operator_arity)

        stds = [max(float(np.std(rv)), 1e-8) for rv in raw_vals_list]

        if len(scale_terms) >= 2:
            sigma_base = float(np.exp(np.mean(np.log(stds))))
        elif len(scale_terms) == 1:
            sigma_others = max(float(np.std(others_val)), 1e-8)
            sigma_base = sigma_others
        else:
            return others_val

        result = others_val.copy()
        for i, (c_node, _, eff_sign) in enumerate(scale_terms):
            c = sigma_base / stds[i] * float(np.exp(np.random.normal(0.0, 0.25)))
            if np.random.rand() < 0.5:
                c = -c
            constants[c_node["const_idx"]] = c
            result = result + eff_sign * c * raw_vals_list[i]
        return result

    def _set_summand_scales(
        self, a_vals: np.ndarray, b_vals: np.ndarray
    ) -> tuple[float, float]:
        """Jointly sample scale constants for two summands with a shared base magnitude and random imbalance."""
        sigma_a = max(float(np.std(a_vals)), 1e-8)
        sigma_b = max(float(np.std(b_vals)), 1e-8)
        sigma_base = np.sqrt(sigma_a * sigma_b) * float(np.exp(np.random.normal(0.0, 0.4)))
        log_ratio = float(np.random.normal(0.0, 0.7))
        c0 = float(sigma_base * np.exp(log_ratio / 2.0) / sigma_a)
        c1 = float(sigma_base * np.exp(-log_ratio / 2.0) / sigma_b)
        if np.random.rand() < 0.5:
            c0 = -c0
        if np.random.rand() < 0.5:
            c1 = -c1
        return c0, c1

    def _fill_and_evaluate(
        self,
        node: dict[str, Any],
        x_data: np.ndarray,
        variables: list[str],
        constants: list,
        operator_arity: dict[str, int],
    ) -> np.ndarray:
        n = len(x_data)
        t = node["type"]

        if t == "var":
            return x_data[:, variables.index(node["name"])]

        if t == "number":
            return np.full(n, node["value"], dtype=np.float64)

        if t == "const":
            ci = node["const_idx"]
            c = float(np.random.normal(0.0, self.cfg["binary_const_std"]))
            constants[ci] = c
            return np.full(n, c, dtype=np.float64)

        op = node["name"]
        arity = node["arity"]

        if arity == 1:
            child = node["children"][0]
            if op == "neg":
                child_vals = self._fill_and_evaluate(child, x_data, variables, constants, operator_arity)
            elif self._is_affine_wrapper(child):
                if self._is_scale_wrapper(child["children"][0]):
                    scale_node, c1_node = child["children"][0], child["children"][1]
                else:
                    scale_node, c1_node = child["children"][1], child["children"][0]
                if scale_node["children"][0]["type"] == "const":
                    c0_node, raw_subtree = scale_node["children"][0], scale_node["children"][1]
                else:
                    c0_node, raw_subtree = scale_node["children"][1], scale_node["children"][0]
                raw_vals = self._fill_and_evaluate(raw_subtree, x_data, variables, constants, operator_arity)
                c0, c1 = self._sample_op_constants(op, raw_vals)
                constants[c0_node["const_idx"]] = c0
                constants[c1_node["const_idx"]] = c1
                child_vals = c0 * raw_vals + c1
            elif self._is_scale_wrapper(child):
                if child["children"][0]["type"] == "const":
                    c0_node, raw_subtree = child["children"][0], child["children"][1]
                else:
                    c0_node, raw_subtree = child["children"][1], child["children"][0]
                raw_vals = self._fill_and_evaluate(raw_subtree, x_data, variables, constants, operator_arity)
                c0, _ = self._sample_op_constants(op, raw_vals, scale_only=True)
                constants[c0_node["const_idx"]] = c0
                child_vals = c0 * raw_vals
            elif self._is_shift_wrapper(child):
                if child["children"][0]["type"] == "const":
                    c1_node, raw_subtree = child["children"][0], child["children"][1]
                else:
                    c1_node, raw_subtree = child["children"][1], child["children"][0]
                raw_vals = self._fill_and_evaluate(raw_subtree, x_data, variables, constants, operator_arity)
                _, c1 = self._sample_op_constants(op, raw_vals, shift_only=True)
                constants[c1_node["const_idx"]] = c1
                child_vals = raw_vals + c1
            elif self._is_summand_pair_with_bias(child):
                if self._is_summand_pair(child["children"][0]):
                    pair_node, c2_node = child["children"][0], child["children"][1]
                else:
                    pair_node, c2_node = child["children"][1], child["children"][0]
                l_c, l_sub = self._scale_parts(pair_node["children"][0])
                r_c, r_sub = self._scale_parts(pair_node["children"][1])
                l_raw = self._fill_and_evaluate(l_sub, x_data, variables, constants, operator_arity)
                r_raw = self._fill_and_evaluate(r_sub, x_data, variables, constants, operator_arity)
                c0, c1 = self._set_summand_scales_for_op(op, l_raw, r_raw)
                constants[l_c["const_idx"]] = c0
                constants[r_c["const_idx"]] = c1
                pair_vals = c0 * l_raw + (-1 if pair_node["name"] == "-" else 1) * c1 * r_raw
                _, c2 = self._sample_op_constants(op, pair_vals, shift_only=True)
                constants[c2_node["const_idx"]] = c2
                child_vals = pair_vals + c2
            elif self._is_summand_pair(child):
                l_c, l_sub = self._scale_parts(child["children"][0])
                r_c, r_sub = self._scale_parts(child["children"][1])
                l_raw = self._fill_and_evaluate(l_sub, x_data, variables, constants, operator_arity)
                r_raw = self._fill_and_evaluate(r_sub, x_data, variables, constants, operator_arity)
                c0, c1 = self._set_summand_scales_for_op(op, l_raw, r_raw)
                constants[l_c["const_idx"]] = c0
                constants[r_c["const_idx"]] = c1
                child_vals = c0 * l_raw + (-1 if child["name"] == "-" else 1) * c1 * r_raw
            else:
                child_vals = self._fill_and_evaluate(child, x_data, variables, constants, operator_arity)
            return self._apply_op(op, child_vals)

        left, right = node["children"][0], node["children"][1]
        l_const = left["type"] == "const"
        r_const = right["type"] == "const"

        if l_const and not r_const:
            if op == "/":
                if self._is_shift_wrapper(right):
                    shift_node, r_raw_node = (
                        (right["children"][0], right["children"][1])
                        if right["children"][0]["type"] == "const"
                        else (right["children"][1], right["children"][0])
                    )
                    r_raw = self._fill_and_evaluate(r_raw_node, x_data, variables, constants, operator_arity)
                    shift = self._sample_denominator_shift(r_raw)
                    constants[shift_node["const_idx"]] = shift
                    r_vals = r_raw + shift
                else:
                    r_vals = self._fill_and_evaluate(right, x_data, variables, constants, operator_arity)
            else:
                r_vals = self._fill_and_evaluate(right, x_data, variables, constants, operator_arity)
            c = float(np.random.normal(0.0, self.cfg["binary_const_std"]))
            constants[left["const_idx"]] = c
            l_vals = np.full(n, c, dtype=np.float64)

        elif r_const and not l_const:
            l_vals = self._fill_and_evaluate(left, x_data, variables, constants, operator_arity)
            if op == "/":
                c = self._sample_denominator_shift(np.zeros(1), allow_straddle=False)
            else:
                c = float(np.random.normal(0.0, self.cfg["binary_const_std"]))
            constants[right["const_idx"]] = c
            r_vals = np.full(n, c, dtype=np.float64)

        elif l_const and r_const:
            c_l = float(np.random.normal(0.0, self.cfg["binary_const_std"]))
            c_r = float(np.random.normal(0.0, self.cfg["binary_const_std"]))
            constants[left["const_idx"]] = c_l
            constants[right["const_idx"]] = c_r
            l_vals = np.full(n, c_l, dtype=np.float64)
            r_vals = np.full(n, c_r, dtype=np.float64)

        else:
            if op == "/":
                l_vals = self._fill_and_evaluate(left, x_data, variables, constants, operator_arity)
                if self._is_shift_wrapper(right):
                    shift_node, r_raw_node = (
                        (right["children"][0], right["children"][1])
                        if right["children"][0]["type"] == "const"
                        else (right["children"][1], right["children"][0])
                    )
                    r_raw = self._fill_and_evaluate(r_raw_node, x_data, variables, constants, operator_arity)
                    shift = self._sample_denominator_shift(r_raw)
                    constants[shift_node["const_idx"]] = shift
                    r_vals = r_raw + shift
                else:
                    r_vals = self._fill_and_evaluate(right, x_data, variables, constants, operator_arity)
            elif op in ("+", "-"):
                return self._fill_sum_chain_jointly(node, x_data, variables, constants, operator_arity)
            else:
                l_vals = self._fill_and_evaluate(left, x_data, variables, constants, operator_arity)
                r_vals = self._fill_and_evaluate(right, x_data, variables, constants, operator_arity)

        return self._apply_op(op, l_vals, r_vals)

    def sample_and_evaluate(
        self,
        skeleton: tuple | list,
        x_data: np.ndarray,
        variables: list[str],
        operator_arity: dict[str, int],
    ) -> tuple[list[float], np.ndarray]:
        """Fill all ``<constant>`` placeholders and return ``(constants, f(x_data))``."""
        n_constants = sum(1 for t in skeleton if t == "<constant>")
        constants: list = [None] * n_constants
        tree = self._parse_prefix(skeleton, set(variables), operator_arity)
        y_values = self._fill_and_evaluate(
            tree, x_data, variables, constants, operator_arity
        )
        return constants, y_values
