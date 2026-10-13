"""Stateless expression-parsing helpers used by RASR."""

from __future__ import annotations

from typing import Callable

import numpy as np
import simplipy as sp


def is_numeric_string(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def required_variable_count(tokens: list[str]) -> int:
    """Return the highest xi index referenced in the token list."""
    max_idx = 0
    for token in tokens:
        if len(token) >= 2 and token[0] == "x" and token[1:].isdigit():
            max_idx = max(max_idx, int(token[1:]))
    return max_idx


def parse_prefix_tokens(
    tokens: list[str],
    operator_arity: dict,
    variables: set[str],
    digit_tokens: set[str],
    exponent_tokens: frozenset[str] = frozenset(),
    constant_token: str | None = "<constant>",
) -> list[str] | None:
    """Parse a decoded token stream into one valid prefix expression, or ``None`` if none exists."""

    def parse_expression_from(start_index: int) -> list[tuple[list[str], int]]:
        if start_index >= len(tokens):
            return []
        current_token = tokens[start_index]

        if constant_token is not None and current_token == constant_token:
            return [([current_token], start_index + 1)]

        if current_token in operator_arity:
            arity = int(operator_arity[current_token])
            partial_parses: list[tuple[list[str], int]] = [([], start_index + 1)]
            for _ in range(arity):
                extended_parses: list[tuple[list[str], int]] = []
                for parsed_children_prefix, next_index in partial_parses:
                    for child_expression, child_end_index in parse_expression_from(next_index):
                        extended_parses.append((parsed_children_prefix + child_expression, child_end_index))
                partial_parses = extended_parses
                if not partial_parses:
                    return []
            parsed_operator_nodes = [
                ([current_token] + parsed_children, end_index)
                for parsed_children, end_index in partial_parses
            ]

            # Canonicalize unary negation over numeric leaves to a signed constant.
            if current_token == "neg" and arity == 1:
                folded: list[tuple[list[str], int]] = []
                for parsed_expression, end_index in parsed_operator_nodes:
                    if len(parsed_expression) == 2 and is_numeric_string(parsed_expression[1]):
                        folded.append(([str(-float(parsed_expression[1]))], end_index))
                    else:
                        folded.append((parsed_expression, end_index))
                return folded

            return parsed_operator_nodes

        if current_token in variables:
            return [([current_token], start_index + 1)]

        # Greedy parsing is unambiguous since a digit never follows a constant.
        mantissa_digit_list: list[str] = []
        scan_index = start_index
        while scan_index < len(tokens) and tokens[scan_index] in digit_tokens and tokens[scan_index].isdigit():
            mantissa_digit_list.append(tokens[scan_index])
            scan_index += 1
        if not mantissa_digit_list:
            return []

        k = len(mantissa_digit_list)
        mantissa_val = int("".join(mantissa_digit_list)) / (10 ** (k - 1))
        end_index = scan_index
        if (
            end_index < len(tokens)
            and tokens[end_index] in exponent_tokens
            and end_index + 1 < len(tokens)
            and tokens[end_index + 1].isdigit()
        ):
            exp_sign = tokens[end_index]
            exp_val = int(tokens[end_index + 1]) if exp_sign == "e+" else -int(tokens[end_index + 1])
            mantissa_val = mantissa_val * (10.0 ** exp_val)
            end_index += 2
        return [([str(mantissa_val)], end_index)]

    candidate_parses = parse_expression_from(0)
    for parsed_expression, parse_end_index in candidate_parses:
        if parse_end_index == len(tokens):
            return parsed_expression
    return None


def expression_to_lambda(
    expression_tokens: list[str],
    simplipy_engine: sp.SimpliPyEngine,
    all_variables: list[str],
) -> tuple[Callable[[np.ndarray], np.ndarray], list[str]]:
    """Compile a prefix expression into a lambda, returning it with the variables it uses."""
    variables = [v for v in all_variables if v in expression_tokens]
    if not variables:
        raise ValueError("Expression has no variable tokens.")
    executable = simplipy_engine.operators_to_realizations(expression_tokens)
    code_string = simplipy_engine.prefix_to_infix(executable, realization=True)
    code = sp.codify(code_string, variables)
    fn = simplipy_engine.code_to_lambda(code)
    return fn, variables
