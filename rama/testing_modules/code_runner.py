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


def _compare_output(actual: str, expected) -> bool:
    """Compare actual stdout to expected value.
    
    Uses sorted comparison for list outputs to handle order-independent tests.
    """
    import ast
    actual = actual.strip()
    if isinstance(expected, str):
        # Try to parse both as Python literals for list/set comparison
        try:
            exp_parsed = ast.literal_eval(expected)
            act_parsed = ast.literal_eval(actual)
            if isinstance(exp_parsed, (list, set, tuple)):
                if isinstance(act_parsed, (list, set, tuple)):
                    if len(exp_parsed) != len(act_parsed):
                        return False
                    # For lists of dicts, compare as multisets using counts
                    try:
                        return sorted(exp_parsed) == sorted(act_parsed)
                    except TypeError:
                        # Fallback: count occurrences of each element as string
                        from collections import Counter
                        return Counter(str(x) for x in exp_parsed) == Counter(str(x) for x in act_parsed)
        except (ValueError, SyntaxError):
            pass
        return actual == expected
    return actual == str(expected)


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

    if not test_cases:
        return {"passed": False, "correct": 0, "total": 0, "details": []}

    correct = 0
    details = []

    for i, case in enumerate(test_cases):
        args = case["inputs"]
        expected = case["expected"]

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
            transformed_args = [f"{arg_transform}({_repr_arg(args[0])})"] + [_repr_arg(a) for a in args[1:]]
            args_str = ", ".join(transformed_args)
            call = f"print({func_name}({args_str}))"

        script = setup + "\n" + code + "\n" + call

        # Check if expected is an exception class or string name
        known_exceptions = {"ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError", "AttributeError"}
        is_exception_expected = (isinstance(expected, type) and issubclass(expected, BaseException)) or \
                                (isinstance(expected, str) and expected in known_exceptions)
        
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
                    details.append({
                        "case": i,
                        "passed": True,
                        "expected": f"raises {expected}",
                        "actual": "raised exception",
                    })
                else:
                    details.append({
                        "case": i,
                        "passed": False,
                        "error": result.stderr.strip()[:200],
                    })
                continue

            output = result.stdout
            if is_exception_expected:
                # Expected exception but function didn't raise one
                passed = False
            else:
                passed = _compare_output(output, expected)
                if passed:
                    correct += 1

            details.append({
                "case": i,
                "passed": passed,
                "expected": str(expected),
                "actual": output.strip()[:200],
            })
        except subprocess.TimeoutExpired:
            details.append({
                "case": i,
                "passed": False,
                "error": "timeout",
            })
        except Exception as e:
            details.append({
                "case": i,
                "passed": False,
                "error": str(e)[:200],
            })

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

        actual_lines = result.stdout.strip().split("\n")
        correct = 0
        details = []

        for i, expected in enumerate(expected_output):
            if i < len(actual_lines):
                passed = _compare_output(actual_lines[i], expected)
                if passed:
                    correct += 1
                details.append({
                    "case": i,
                    "passed": passed,
                    "expected": expected,
                    "actual": actual_lines[i].strip()[:200],
                })
            else:
                details.append({
                    "case": i,
                    "passed": False,
                    "expected": expected,
                    "actual": "<missing>",
                })

        total = len(expected_output)
        return {
            "passed": correct == total and total > 0,
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

    Uses test_cases for simple function tests, test_script for class-based tests.
    """
    if "test_script" in task:
        return run_class_task(code, task)
    elif "test_cases" in task:
        return run_code_task(code, task)
    else:
        return {"passed": False, "correct": 0, "total": 0, "details": []}
