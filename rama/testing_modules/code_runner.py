"""Generic code task runner for data-driven testing.

Reads test tasks from JSON files and runs LLM-generated code against
input/output test cases. Replaces the old hardcoded _check_* functions.
"""

import json
import subprocess
import sys
from pathlib import Path

PYTHON_BIN = sys.executable or "python"


def load_task(path: Path) -> dict:
    """Load a test task from a JSON file."""
    with open(path) as f:
        return json.load(f)


def _compare_output(
    actual: str, expected, *, order_independent: bool = False
) -> bool:
    """Compare actual stdout to expected value.

    List order is significant unless a fixture explicitly opts out with
    ``order_independent``.  Several coding tasks specifically assess ordering,
    so treating every list as a set makes correct-looking scores meaningless.
    """
    import ast

    actual = actual.strip()

    try:
        actual_value = ast.literal_eval(actual)
    except (ValueError, SyntaxError):
        actual_value = actual

    if isinstance(expected, str):
        try:
            expected_value = ast.literal_eval(expected)
        except (ValueError, SyntaxError):
            expected_value = expected
    else:
        expected_value = expected

    if not order_independent:
        return actual_value == expected_value

    if isinstance(expected_value, (list, tuple)) and isinstance(
        actual_value, (list, tuple)
    ):
        if len(expected_value) != len(actual_value):
            return False
        # repr preserves the order of nested values while comparing the outer
        # collection as a multiset, which is what these fixtures opt into.
        from collections import Counter

        return Counter(map(repr, expected_value)) == Counter(map(repr, actual_value))

    return actual_value == expected_value


def _validate_output(actual: str, args: list, validator: str) -> bool:
    """Validate outputs whose contract permits more than one exact value."""
    import ast

    try:
        value = ast.literal_eval(actual.strip())
    except (ValueError, SyntaxError):
        return False

    if validator == "two_sum_indices":
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(type(index) is not int for index in value)
        ):
            return False
        first, second = value
        numbers, target = args
        return (
            first != second
            and 0 <= first < len(numbers)
            and 0 <= second < len(numbers)
            and numbers[first] + numbers[second] == target
        )

    raise ValueError(f"Unknown fixture output validator: {validator}")


def _repr_arg(arg) -> str:
    """Convert a JSON-serializable arg to Python source code."""
    if arg is None:
        return "None"
    if isinstance(arg, bool):
        return "True" if arg else "False"
    if isinstance(arg, int):
        return str(arg)
    if isinstance(arg, float):
        return repr(arg)
    if isinstance(arg, str):
        return repr(arg)
    if isinstance(arg, list):
        items = ", ".join(_repr_arg(a) for a in arg)
        return f"[{items}]"
    if isinstance(arg, dict):
        pairs = ", ".join(f"{_repr_arg(k)}: {_repr_arg(v)}" for k, v in arg.items())
        return f"{{{pairs}}}"
    return repr(arg)


def run_code_task(code: str, task: dict) -> dict:
    """Run a simple function code task against test cases.

    Args:
        code: The LLM-generated code (function definition only)
        task: Task dict with function_name, setup_code, test_cases,
              optional post_process function name for output transformation

    Returns:
        dict with passed, correct, total, details
    """
    func_name = task["function_name"]
    setup = task.get("setup_code", "") or ""
    test_cases = task.get("test_cases", [])
    post_process = task.get("post_process", "")
    task_order_independent = bool(task.get("order_independent", False))
    task_validator = task.get("validator")

    if not test_cases:
        return {"passed": False, "correct": 0, "total": 0, "details": []}

    correct = 0
    details = []

    for i, case in enumerate(test_cases):
        args = case["inputs"]
        expected = case["expected"]
        order_independent = bool(
            case.get("order_independent", task_order_independent)
        )
        validator = case.get("validator", task_validator)

        # Build argument string for function call
        args_str = ", ".join(_repr_arg(a) for a in args)

        # Build call with optional post_process
        if post_process:
            call = f"print({post_process}({func_name}({args_str})))"
        else:
            call = f"print({func_name}({args_str}))"

        # Handle arg_transform: apply transform function to first argument
        arg_transform = task.get("arg_transform", "")
        if arg_transform and not post_process:
            # Transform the first argument
            transformed_args = [f"{arg_transform}({_repr_arg(args[0])})"] + [
                _repr_arg(a) for a in args[1:]
            ]
            args_str = ", ".join(transformed_args)
            call = f"print({func_name}({args_str}))"

        script = setup + "\n" + code + "\n" + call

        # Check if expected is an exception class or string name
        known_exceptions = {
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "RuntimeError",
            "AttributeError",
        }
        is_exception_expected = (
            isinstance(expected, type) and issubclass(expected, BaseException)
        ) or (isinstance(expected, str) and expected in known_exceptions)

        try:
            result = subprocess.run(
                [PYTHON_BIN, "-c", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                if is_exception_expected:
                    # Function raised an exception as expected
                    passed = True
                    correct += 1
                    details.append(
                        {
                            "case": i,
                            "passed": True,
                            "expected": f"raises {expected}",
                            "actual": "raised exception",
                        }
                    )
                else:
                    details.append(
                        {
                            "case": i,
                            "passed": False,
                            "error": result.stderr.strip()[:200],
                        }
                    )
                continue

            output = result.stdout
            if is_exception_expected:
                # Expected exception but function didn't raise one
                passed = False
            elif validator:
                passed = _validate_output(output, args, validator)
            else:
                passed = _compare_output(
                    output, expected, order_independent=order_independent
                )

            if passed:
                correct += 1

            details.append(
                {
                    "case": i,
                    "passed": passed,
                    "expected": str(expected),
                    "actual": output.strip()[:200],
                }
            )
        except subprocess.TimeoutExpired:
            details.append(
                {
                    "case": i,
                    "passed": False,
                    "error": "timeout",
                }
            )
        except Exception as e:
            details.append(
                {
                    "case": i,
                    "passed": False,
                    "error": str(e)[:200],
                }
            )

    total = len(test_cases)
    return {
        "passed": correct == total and total > 0,
        "correct": correct,
        "total": total,
        "details": details,
    }


def run_class_task(code: str, task: dict) -> dict:
    """Run a class-based code task with a test script.

    Args:
        code: The LLM-generated code (class definition only)
        task: Task dict with function_name, setup_code, test_script, expected_output

    Returns:
        dict with passed, correct, total, details
    """
    setup = task.get("setup_code", "") or ""
    test_script = task.get("test_script", "")
    expected_output = task.get("expected_output", [])
    order_independent = bool(task.get("order_independent", False))

    if not test_script or not expected_output:
        return {"passed": False, "correct": 0, "total": 0, "details": []}

    script = setup + "\n" + code + "\n" + test_script

    try:
        result = subprocess.run(
            [PYTHON_BIN, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {
                "passed": False,
                "correct": 0,
                "total": len(expected_output),
                "details": [{"error": result.stderr.strip()[:500]}],
            }

        # splitlines keeps an intentional final blank result (for example a
        # valid empty-string return) instead of stripping it away.
        actual_lines = result.stdout.splitlines()
        correct = 0
        details = []

        for i, expected in enumerate(expected_output):
            if i < len(actual_lines):
                passed = _compare_output(
                    actual_lines[i], expected, order_independent=order_independent
                )
                if passed:
                    correct += 1
                details.append(
                    {
                        "case": i,
                        "passed": passed,
                        "expected": expected,
                        "actual": actual_lines[i].strip()[:200],
                    }
                )
            else:
                details.append(
                    {
                        "case": i,
                        "passed": False,
                        "expected": expected,
                        "actual": "<missing>",
                    }
                )

        if len(actual_lines) != len(expected_output):
            details.append(
                {
                    "error": (
                        "output line count mismatch: "
                        f"expected {len(expected_output)}, got {len(actual_lines)}"
                    )
                }
            )

        total = len(expected_output)
        return {
            "passed": (
                correct == total
                and total > 0
                and len(actual_lines) == len(expected_output)
            ),
            "correct": correct,
            "total": total,
            "details": details,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "correct": 0,
            "total": len(expected_output),
            "details": [{"error": "timeout"}],
        }
    except Exception as e:
        return {
            "passed": False,
            "correct": 0,
            "total": len(expected_output),
            "details": [{"error": str(e)[:500]}],
        }


def run_task(code: str, task: dict) -> dict:
    """Run a code task, auto-detecting simple vs class-based.

    Runs every declared test mode.  Some fixtures have both direct cases and a
    script to cover function-level and integration-style behavior; neither
    source of truth should be silently discarded.
    """
    results = []
    if task.get("test_script"):
        results.append(("test_script", run_class_task(code, task)))
    if task.get("test_cases"):
        results.append(("test_cases", run_code_task(code, task)))

    if not results:
        return {"passed": False, "correct": 0, "total": 0, "details": []}

    if len(results) == 1:
        return results[0][1]

    details = []
    correct = 0
    total = 0
    for source, result in results:
        correct += result["correct"]
        total += result["total"]
        details.extend({"source": source, **detail} for detail in result["details"])

    return {
        "passed": all(result["passed"] for _, result in results),
        "correct": correct,
        "total": total,
        "details": details,
    }
