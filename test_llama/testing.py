import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _resolve_python_bin() -> str:
    """Prefer the repo virtualenv interpreter when available."""
    current = Path(sys.executable) if sys.executable else None
    if current and ".venv" in current.parts:
        return str(current)

    repo_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    if repo_python.exists():
        return str(repo_python)

    return sys.executable or "python3"


PYTHON_BIN = _resolve_python_bin()


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    category: str
    prompt: str
    response: str
    ran: bool = False
    correct: bool = False
    score: int = 0


@dataclass
class TestSuiteResult:
    """Result of running test suite."""

    model_name: str
    backend: str
    results: list[TestResult]
    overall_score: float


def get_tests_dir() -> Path:
    """Get the tests directory."""
    return Path(__file__).parent.parent / "tests"


def discover_tests() -> dict[str, list[Path]]:
    """Discover all test files by category."""
    tests_dir = get_tests_dir()
    categories = {}

    for category in ["code", "debugging"]:
        cat_dir = tests_dir / category
        if cat_dir.exists():
            categories[category] = sorted(cat_dir.glob("*.txt"))

    return categories


def run_chat_completion(
    port: int,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    seed: int = 42,
) -> str:
    """Call the model via llama.cpp API."""
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except urllib.error.HTTPError as e:
        return f"ERROR: {e.code} {e.reason}"
    except Exception as e:
        return f"ERROR: {str(e)}"


def score_response(judge_port: int, test_name: str, prompt: str, response: str) -> int:
    """Use LLM as judge to score response 1-10."""
    judge_prompt = f"""You are an impartial judge.
    Score the following model response on a scale of 1-10.

TEST: {test_name}
PROMPT: {prompt[:500]}...
RESPONSE: {response[:1000]}

Score based on:
- Correctness and completeness
- Code quality and runnability
- Following instructions

Return ONLY a number from 1-10. No explanation."""

    result = run_chat_completion(
        judge_port, judge_prompt, temperature=0.0, max_tokens=5, seed=42
    )

    # Extract number
    import re

    match = re.search(r"\d+", result)
    if match:
        return min(10, max(1, int(match.group())))
    return 5


def test_code(response: str) -> tuple[bool, str]:
    """Test Python code by compiling the extracted snippet.

    Returns: (compiles_successfully, output)
    """
    code = _extract_code(response)

    if not code or ("def " not in code and "class " not in code):
        return False, "No code found"

    try:
        compile(code, "<submission>", "exec")
        return True, ""
    except Exception as e:
        return False, str(e)


def _normalize_code(code: str) -> str:
    """Normalize extracted code by removing stray markdown fence lines."""
    if not code:
        return ""

    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if lines and lines[0].strip().lower() in {"python", "py"}:
        lines.pop(0)

    cleaned = [line for line in lines if not line.strip().startswith("```")]

    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned).strip()


def _extract_code(response: str) -> str:
    """Extract Python code from response."""
    import re

    # Try fenced code blocks first.
    fenced_blocks = []
    for match in re.finditer(
        r"```([^\n`]*)\n?(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE
    ):
        language = (match.group(1) or "").strip().lower()
        body = match.group(2)
        fenced_blocks.append((language, body))

    if fenced_blocks:
        preferred_languages = {
            "python",
            "py",
            "python3",
            "",
        }
        for language, body in fenced_blocks:
            if language in preferred_languages:
                return _normalize_code(body)
        return _normalize_code(fenced_blocks[0][1])

    lines = response.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ", "class ", "def ", "@")):
            return _normalize_code("\n".join(lines[i:]))

    return _normalize_code(response)


def _evaluate_single_response(
    test_name: str, category: str, response: str
) -> tuple[bool, bool, int, str]:
    """Evaluate one response with the same logic used by the live test runner."""
    ran = False
    correct = False
    score = 0
    output = ""

    if category not in ["code", "debugging"]:
        return ran, correct, score, output

    if response.startswith("ERROR:"):
        return False, False, 0, response

    ran, output = test_code(response)
    if not ran:
        return ran, correct, score, output

    if category == "code":
        test_fn = {
            "inventory_service": _check_inventory_service,
            "job_queue_service": _check_job_queue_service,
            "rate_limiter": _check_rate_limiter,
            "query_filter_parser": _check_query_filter_parser,
            "process_logs": _check_process_logs,
            "shortest_path_with_break": _check_shortest_path_with_break,
            "arithmetic_evaluator": _check_arithmetic_evaluator,
            "merge_k_lists": _check_merge_k_lists,
            "word_ladder": _check_word_ladder,
            "min_window": _check_min_window,
            "regex_matching": _check_regex_match,
            "sudoku_solver": _check_sudoku_solver,
            "median_two_sorted_arrays": _check_median_two_sorted_arrays,
            "substring_concatenation": _check_substring_concatenation,
            "remove_invalid_parentheses": _check_remove_invalid_parentheses,
            "wildcard_matching": _check_wildcard_matching,
            "distinct_subsequences": _check_distinct_subsequences,
            "top_k_frequent": _check_top_k_frequent,
            "spiral_matrix": _check_spiral_matrix,
            "group_anagrams": _check_group_anagrams,
            "expression_add_operators": _check_expression_add_operators,
            "smallest_range_k_lists": _check_smallest_range_k_lists,
            "alien_dictionary_order": _check_alien_dictionary_order,
            "candy_crush_board": _check_candy_crush_board,
            "word_search_ii": _check_word_search_ii,
        }.get(test_name)
        if test_fn:
            correct = test_fn(response)
            score = 10 if correct else 0
    else:
        code = _extract_code(response)
        num_correct, total_tests = _check_debug_test(code, test_name)
        correct = total_tests > 0 and num_correct == total_tests
        score = 0 if total_tests == 0 else (10 * num_correct) // total_tests

    return ran, correct, score, output


def _check_merge_k_lists(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def merge_k_lists", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
# Helper to build list
def build(vals):
    head = None
    for v in reversed(vals):
        head = ListNode(v, head)
    return head

def to_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

# Test: [1->4->5, 1->3->4, 2->6]
lists = [
    build([1, 4, 5]),
    build([1, 3, 4]),
    build([2, 6])
]
merged = merge_k_lists(lists)
print(to_list(merged))

# Edge cases
print(to_list(build([])))  # empty returns empty
print(to_list(merge_k_lists([])))  # no lists
print(to_list(merge_k_lists([build([]), build([1])])))
print(to_list(merge_k_lists([build([0]), build([0])])))
print(to_list(merge_k_lists([build([-3, -1, 2]), build([-2, 2, 3]), build([])])))
print(to_list(merge_k_lists([build([1, 1]), build([]), build([1]), build([-1, 0, 2]), build([])])))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.strip().split("\n")
        expected = [
            "[1, 1, 2, 3, 4, 4, 5, 6]",
            "[]",
            "[]",
            "[1]",
            "[0, 0]",
            "[-3, -2, -1, 2, 2, 3]",
            "[-1, 0, 1, 1, 1, 2]",
        ]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_word_ladder(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def ladder_length", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
print(ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
print(ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))
print(ladder_length("a", "c", ["a", "b", "c"]))
print(ladder_length("lost", "cost", ["most", "fost", "cost", "lost"]))
print(ladder_length("talk", "tail", ["tall", "tail", "balk", "bail"]))
print(ladder_length("red", "tax", ["ted", "tex", "red", "tax", "tad", "den", "rex", "pee"]))
print(ladder_length("same", "same", ["same", "came", "lame"]))
print(ladder_length("cold", "warm", ["cord", "card", "ward", "warm", "wold", "wald", "worm"]))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.strip().split("\n")
        expected = ["5", "0", "2", "2", "3", "4", "1", "5"]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_min_window(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def min_window", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
print(min_window("ADOBECODEBANC", "ABC"))
print(min_window("a", "a"))
print(min_window("a", "aa"))
print(min_window("aa", "aa"))
print(min_window("bba", "ab"))
print(min_window("cabwefgewcwaefgcf", "cae"))
print(min_window("anything", ""))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = ["BANC", "a", "", "aa", "ba", "cwae", ""]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_regex_match(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def is_match", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
print(is_match("aa", "a"))
print(is_match("aa", "a*"))
print(is_match("ab", ".*"))
print(is_match("aab", "c*a*b"))
print(is_match("mississippi", "mis*is*p*."))
print(is_match("ab", ".*c"))
print(is_match("aaa", "ab*a*c*a"))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = ["False", "True", "True", "True", "False", "False", "True"]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_sudoku_solver(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def solve_sudoku", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
solved = [
    ["5", "3", "4", "6", "7", "8", "9", "1", "2"],
    ["6", "7", "2", "1", "9", "5", "3", "4", "8"],
    ["1", "9", "8", "3", "4", "2", "5", "6", "7"],
    ["8", "5", "9", "7", "6", "1", "4", "2", "3"],
    ["4", "2", "6", "8", "5", "3", "7", "9", "1"],
    ["7", "1", "3", "9", "2", "4", "8", "5", "6"],
    ["9", "6", "1", "5", "3", "7", "2", "8", "4"],
    ["2", "8", "7", "4", "1", "9", "6", "3", "5"],
    ["3", "4", "5", "2", "8", "6", "1", "7", "9"],
]

def blank(board, row, col):
    puzzle = [r[:] for r in board]
    puzzle[row][col] = "."
    return puzzle

puzzles = [
    [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"],
    ],
    blank(solved, 0, 2),
    blank(solved, 4, 4),
    blank(solved, 8, 8),
    blank(solved, 6, 6),
]
for board in puzzles:
    solve_sudoku(board)
    print(board == solved)
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=20
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = ["True", "True", "True", "True", "True"]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_median_two_sorted_arrays(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def find_median_sorted_arrays", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    ([1, 3], [2]),
    ([1, 2], [3, 4]),
    ([], [1]),
    ([2], []),
    ([0, 0], [0, 0]),
    ([1], [2, 3, 4, 5, 6]),
    ([-5, -3, -1], [-2, 4, 8, 10]),
    ([1], [2]),
    ([1, 2, 3], [4, 5, 6, 7]),
    ([50], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ([100, 101], [1, 2, 3, 4, 5, 6]),
    ([-10, -5], [-4, -3, -2, -1, 0]),
]
for nums1, nums2 in cases:
    print(find_median_sorted_arrays(nums1, nums2))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            2.0,
            2.5,
            1.0,
            2.0,
            0.0,
            3.5,
            -1.0,
            1.5,
            4.0,
            5.5,
            4.5,
            -3.0,
        ]
        if len(lines) != len(expected):
            return False
        return all(
            abs(float(lines[i].strip()) - expected[i]) < 1e-9
            for i in range(len(expected))
        )
    except:
        return False


def _check_substring_concatenation(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def find_substring", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
print(find_substring("barfoothefoobarman", ["foo", "bar"]))
print(find_substring("wordgoodgoodgoodbestword", ["word", "good", "best", "word"]))
print(find_substring("barfoofoobarthefoobarman", ["bar", "foo", "the"]))
print(find_substring("lingmindraboofooowingdingbarrwingmonkeypoundcake", ["fooo", "barr", "wing", "ding", "wing"]))
print(find_substring("", ["foo"]))
print(find_substring("aaaaaa", ["aa", "aa"]))
print(find_substring("aaaaa", ["a", "a", "a"]))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            [0, 9],
            [],
            [6, 9, 12],
            [13],
            [],
            [0, 1, 2],
            [0, 1, 2],
        ]
        if len(lines) != len(expected):
            return False
        return all(
            ast.literal_eval(lines[i].strip()) == expected[i]
            for i in range(len(expected))
        )
    except:
        return False


def _check_remove_invalid_parentheses(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def remove_invalid_parentheses", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
print(sorted(remove_invalid_parentheses("()())()")))
print(sorted(remove_invalid_parentheses("(a)())()")))
print(sorted(remove_invalid_parentheses(")(")))
print(sorted(remove_invalid_parentheses("n")))
print(sorted(remove_invalid_parentheses("(((")))
print(sorted(remove_invalid_parentheses("()v)")))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            ["(())()", "()()()"],
            ["(a())()", "(a)()()"],
            [""],
            ["n"],
            [""],
            ["()v", "(v)"],
        ]
        if len(lines) != len(expected):
            return False
        return all(
            ast.literal_eval(lines[i].strip()) == expected[i]
            for i in range(len(expected))
        )
    except:
        return False


def _check_wildcard_matching(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def is_wildcard_match", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
print(is_wildcard_match("aa", "a"))
print(is_wildcard_match("aa", "*"))
print(is_wildcard_match("cb", "?a"))
print(is_wildcard_match("adceb", "*a*b"))
print(is_wildcard_match("acdcb", "a*c?b"))
print(is_wildcard_match("", "*"))
print(is_wildcard_match("", "?"))
print(is_wildcard_match("abcde", "a*de"))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = ["False", "True", "False", "True", "False", "True", "False", "True"]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_distinct_subsequences(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def num_distinct", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
print(num_distinct("rabbbit", "rabbit"))
print(num_distinct("babgbag", "bag"))
print(num_distinct("", ""))
print(num_distinct("abc", ""))
print(num_distinct("", "a"))
print(num_distinct("aaaaa", "aa"))
print(num_distinct("ABCDE", "ACE"))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = ["3", "5", "1", "1", "0", "10", "1"]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_top_k_frequent(response: str) -> bool:
    code = _extract_code(response)
    import ast
    from collections import Counter
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def top_k_frequent\b", code):
        return False

    def is_valid_top_k(nums, k, actual):
        if not isinstance(actual, (list, tuple)) or len(actual) != k:
            return False
        if len(set(actual)) != len(actual):
            return False

        freq = Counter(nums)
        actual_counts = [freq.get(value, 0) for value in actual]
        if any(count == 0 for count in actual_counts):
            return False

        sorted_counts = sorted(freq.values(), reverse=True)
        threshold = sorted_counts[k - 1] if 0 < k <= len(sorted_counts) else None
        if threshold is None:
            return False

        must_have = {value for value, count in freq.items() if count > threshold}
        if not must_have.issubset(set(actual)):
            return False

        allowed = {value for value, count in freq.items() if count >= threshold}
        return set(actual).issubset(allowed)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    ([1, 1, 1, 2, 2, 3], 2),
    ([4, 4, 1, 1, 2, 2], 2),
    ([5], 1),
    ([3, 3, 2, 2, 1], 2),
    ([1, 2, 2, 3, 3, 3], 2),
    ([-1, -1, -2, -2, -2, 0], 3),
    ([9, 9, 9, 8, 8, 7, 6, 6, 6, 6], 2),
]
for nums, k in cases:
    print(top_k_frequent(nums, k))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            ([1, 1, 1, 2, 2, 3], 2),
            ([4, 4, 1, 1, 2, 2], 2),
            ([5], 1),
            ([3, 3, 2, 2, 1], 2),
            ([1, 2, 2, 3, 3, 3], 2),
            ([-1, -1, -2, -2, -2, 0], 3),
            ([9, 9, 9, 8, 8, 7, 6, 6, 6, 6], 2),
        ]
        return len(lines) == len(expected) and all(
            is_valid_top_k(
                expected[i][0],
                expected[i][1],
                ast.literal_eval(lines[i].strip()),
            )
            for i in range(len(expected))
        )
    except:
        return False


def _check_spiral_matrix(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def spiral_order\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[1, 2, 3, 4]],
    [[1], [2], [3]],
    [],
    [[1, 2], [3, 4], [5, 6]],
    [[1, 2], [3, 4]],
    [[1]],
    [[1, 2, 3], [4, 5, 6]],
]
for matrix in cases:
    print(spiral_order(matrix))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            [1, 2, 3, 6, 9, 8, 7, 4, 5],
            [1, 2, 3, 4],
            [1, 2, 3],
            [],
            [1, 2, 4, 6, 5, 3],
            [1, 2, 4, 3],
            [1],
            [1, 2, 3, 6, 5, 4],
        ]
        return len(lines) == len(expected) and all(
            ast.literal_eval(lines[i].strip()) == expected[i]
            for i in range(len(expected))
        )
    except:
        return False


def _check_group_anagrams(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def group_anagrams\b", code):
        return False

    def canonicalize(groups):
        normalized = [sorted(group) for group in groups]
        return sorted(normalized, key=lambda group: tuple(group))

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    ["eat", "tea", "tan", "ate", "nat", "bat"],
    [],
    ["ab", "ba", "abc", "cab"],
    ["x"],
    ["bob", "obb", "boo", "oob", "bob"],
    ["", "", "a", "b", "ab", "ba", ""],
]
for words in cases:
    print(group_anagrams(words))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
            [],
            [["ab", "ba"], ["abc", "cab"]],
            [["x"]],
            [["bob", "obb", "bob"], ["boo", "oob"]],
            [["", "", ""], ["a"], ["ab", "ba"], ["b"]],
        ]
        return len(lines) == len(expected) and all(
            canonicalize(ast.literal_eval(lines[i].strip()))
            == canonicalize(expected[i])
            for i in range(len(expected))
        )
    except:
        return False


def _check_expression_add_operators(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def add_operators\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    ("123", 6),
    ("232", 8),
    ("105", 5),
    ("00", 0),
    ("3456237490", 9191),
]
for num, target in cases:
    print(add_operators(num, target))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=20
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            sorted(["1+2+3", "1*2*3"]),
            sorted(["2*3+2", "2+3*2"]),
            sorted(["1*0+5", "10-5"]),
            sorted(["0+0", "0-0", "0*0"]),
            [],
        ]
        if len(lines) != len(expected):
            return False
        for i, exp in enumerate(expected):
            try:
                got = sorted(ast.literal_eval(lines[i].strip()))
            except:
                return False
            if got != exp:
                return False
        return True
    except:
        return False


def _check_smallest_range_k_lists(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def smallest_range\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]],
    [[1,2,3],[1,2,3],[1,2,3]],
    [[1],[2],[3]],
    [[1,5,8],[4,12],[7,8,10]],
    [[-10,-5,0],[1,2,3],[4,5,6]],
    [[1,4],[2,5],[3,6]],
    [[1,1,2],[1,2,2],[1,2,3]],
]
for nums in cases:
    print(smallest_range(nums))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [[20, 24], [1, 1], [1, 3], [4, 7], [0, 4], [1, 3], [1, 1]]
        if len(lines) != len(expected):
            return False
        for i, exp in enumerate(expected):
            try:
                got = list(ast.literal_eval(lines[i].strip()))
            except:
                return False
            if got != exp:
                return False
        return True
    except:
        return False


def _check_alien_dictionary_order(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def alien_order\b", code):
        return False

    def valid_order(words, order):
        chars = set("".join(words))
        if not order or len(order) != len(chars) or set(order) != chars:
            return False
        pos = {ch: i for i, ch in enumerate(order)}
        for w1, w2 in zip(words, words[1:]):
            if len(w1) > len(w2) and w1.startswith(w2):
                return False
            for a, b in zip(w1, w2):
                if a != b:
                    if pos[a] > pos[b]:
                        return False
                    break
        return True

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    ["wrt", "wrf", "er", "ett", "rftt"],
    ["z", "x"],
    ["z", "x", "z"],
    ["abc", "ab"],
    ["baa", "abcd", "abca", "cab", "cad"],
]
for words in cases:
    print(alien_order(words))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [True, True, False, False, True]
        if len(lines) != len(expected):
            return False
        for i, exp in enumerate(expected):
            got = lines[i].strip()
            if (
                valid_order(
                    [
                        ["wrt", "wrf", "er", "ett", "rftt"],
                        ["z", "x"],
                        ["z", "x", "z"],
                        ["abc", "ab"],
                        ["baa", "abcd", "abca", "cab", "cad"],
                    ][i],
                    got,
                )
                != exp
            ):
                return False
        return True
    except:
        return False


def _check_candy_crush_board(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def candy_crush\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
def run(board):
    b = [row[:] for row in board]
    result = candy_crush(b)
    print(result if result is not None else b)

boards = [
    [[110,5,112,113,114],[210,211,5,213,214],[310,311,3,313,314],[410,411,412,5,414],[5,1,512,3,3],[610,4,1,613,614],[710,1,2,713,714],[810,1,2,1,1],[1,1,2,2,2],[4,1,4,4,1014]],
    [[1,1,1],[2,2,2],[3,3,3]],
    [[1,2,3],[4,5,6],[7,8,9]],
    [[1,1,1,2],[3,4,5,6],[7,8,9,9],[7,7,7,9]],
    [[1,2,3,4],[1,5,6,4],[1,7,8,4],[9,9,9,4]],
]
for board in boards:
    run(board)
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=20
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [110, 0, 0, 0, 114],
                [210, 0, 0, 0, 214],
                [310, 0, 0, 113, 314],
                [410, 0, 0, 213, 414],
                [610, 211, 112, 313, 614],
                [710, 311, 412, 613, 714],
                [810, 411, 512, 713, 1014],
            ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0, 0, 2], [0, 0, 0, 6], [3, 4, 5, 9], [7, 8, 9, 9]],
            [[0, 0, 0, 0], [0, 2, 3, 0], [0, 5, 6, 0], [0, 7, 8, 0]],
        ]
        if len(lines) != len(expected):
            return False
        for i, exp in enumerate(expected):
            try:
                got = ast.literal_eval(lines[i].strip())
            except:
                return False
            if got != exp:
                return False
        return True
    except:
        return False


def _check_word_search_ii(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def find_words\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    (
        [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],
        ["oath","pea","eat","rain"],
    ),
    (
        [["a","a"],["a","a"]],
        ["a","aa","aaa","aaaa","aaaaa"],
    ),
    (
        [["a","b"],["c","d"]],
        ["ab","ac","bd","ca","abdc","dc"],
    ),
    (
        [["a","b"],["c","d"]],
        ["ab","abd","ac","bd","ca","dc","ad","abc","abcd","ab"],
    ),
    (
        [["x"]],
        ["x","xx","y"],
    ),
    (
        [["a","b","c"],["d","e","f"],["g","h","i"]],
        ["abe","cfi","adg","beh","defi","aei"],
    ),
]
for board, words in cases:
    print(find_words(board, words))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            sorted(["oath", "eat"]),
            sorted(["a", "aa", "aaa", "aaaa"]),
            sorted(["ab", "abdc", "ac", "bd", "ca", "dc"]),
            sorted(["ab", "abd", "ac", "bd", "ca", "dc"]),
            sorted(["x"]),
            sorted(["abe", "adg", "beh", "cfi", "defi"]),
        ]
        if len(lines) != len(expected):
            return False
        for i, exp in enumerate(expected):
            try:
                got = sorted(set(ast.literal_eval(lines[i].strip())))
            except:
                return False
            if got != exp:
                return False
        return True
    except:
        return False


def _check_inventory_service(response: str) -> bool:
    code = _extract_code(response)
    import json
    import re
    import tempfile
    from pathlib import Path

    if "FastAPI" not in code or not re.search(r"\bapp\s*=\s*FastAPI\(", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
import inspect
import json
from fastapi import HTTPException

def to_jsonable(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict") and callable(value.dict):
        value = value.dict()

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value

def find_route(path, method):
    for route in app.routes:
        methods = getattr(route, "methods", set())
        if getattr(route, "path", None) == path and method in methods:
            return route
    raise AssertionError(f"missing route {method} {path}")

def make_body(route, **kwargs):
    params = list(inspect.signature(route.endpoint).parameters.values())
    assert params, "route must accept a body parameter"
    annotation = params[-1].annotation
    return annotation(**kwargs)

def expect_status(status_code, fn, *args):
    try:
        fn(*args)
    except HTTPException as exc:
        print(exc.status_code)
        return
    raise AssertionError(f"expected HTTPException {status_code}")

post_items = find_route("/items", "POST")
get_item = find_route("/items/{id}", "GET") if any(getattr(route, "path", None) == "/items/{id}" for route in app.routes) else find_route("/items/{item_id}", "GET")
reserve_item = find_route("/reserve/{id}", "POST") if any(getattr(route, "path", None) == "/reserve/{id}" for route in app.routes) else find_route("/reserve/{item_id}", "POST")
release_item = find_route("/release/{id}", "POST") if any(getattr(route, "path", None) == "/release/{id}" for route in app.routes) else find_route("/release/{item_id}", "POST")
list_items = find_route("/items", "GET")

print(json.dumps(to_jsonable(post_items.endpoint(make_body(post_items, id="b", name="beta", stock=5))), sort_keys=True))
print(json.dumps(to_jsonable(post_items.endpoint(make_body(post_items, id="a", name="alpha", stock=3))), sort_keys=True))
print(json.dumps(to_jsonable(post_items.endpoint(make_body(post_items, id="c", name="gamma", stock=1))), sort_keys=True))
expect_status(400, post_items.endpoint, make_body(post_items, id="a", name="again", stock=2))

print(json.dumps(to_jsonable(get_item.endpoint("a")), sort_keys=True))
print(json.dumps(to_jsonable(reserve_item.endpoint("a", make_body(reserve_item, amount=2))), sort_keys=True))
print(json.dumps(to_jsonable(get_item.endpoint("a")), sort_keys=True))
print(json.dumps(to_jsonable(reserve_item.endpoint("a", make_body(reserve_item, amount=1))), sort_keys=True))
print(json.dumps(to_jsonable(get_item.endpoint("a")), sort_keys=True))
expect_status(400, reserve_item.endpoint, "a", make_body(reserve_item, amount=1))
expect_status(400, release_item.endpoint, "a", make_body(release_item, amount=4))
print(json.dumps(to_jsonable(release_item.endpoint("a", make_body(release_item, amount=2))), sort_keys=True))
print(json.dumps(to_jsonable(get_item.endpoint("a")), sort_keys=True))
print(json.dumps(to_jsonable(reserve_item.endpoint("c", make_body(reserve_item, amount=1))), sort_keys=True))
expect_status(400, reserve_item.endpoint, "c", make_body(reserve_item, amount=1))
print(json.dumps(to_jsonable(release_item.endpoint("c", make_body(release_item, amount=1))), sort_keys=True))
print(json.dumps(to_jsonable(list_items.endpoint()), sort_keys=True))
expect_status(404, get_item.endpoint, "missing")
expect_status(400, reserve_item.endpoint, "a", make_body(reserve_item, amount=0))
expect_status(400, reserve_item.endpoint, "a", make_body(reserve_item, amount=-1))
expect_status(400, release_item.endpoint, "a", make_body(release_item, amount=0))
expect_status(400, release_item.endpoint, "a", make_body(release_item, amount=-1))
"""
            )
            t = f.name

        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=20
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False

        lines = result.stdout.splitlines()
        if len(lines) != 22:
            return False

        item_b = json.loads(lines[0].strip())
        item_a_initial = json.loads(lines[4].strip())
        item_a_reserve_result_2 = json.loads(lines[5].strip())
        item_a_after_reserve_2 = json.loads(lines[6].strip())
        item_a_reserve_result_3 = json.loads(lines[7].strip())
        item_a_after_reserve_3 = json.loads(lines[8].strip())
        item_a_after_release = json.loads(lines[12].strip())
        item_c_after_reserve = json.loads(lines[13].strip())
        item_c_after_release = json.loads(lines[15].strip())
        all_items = json.loads(lines[16].strip())

        return (
            item_b == {"id": "b", "name": "beta", "stock": 5, "reserved": 0}
            and json.loads(lines[1].strip())
            == {"id": "a", "name": "alpha", "stock": 3, "reserved": 0}
            and json.loads(lines[2].strip())
            == {"id": "c", "name": "gamma", "stock": 1, "reserved": 0}
            and lines[3].strip() == "400"
            and item_a_initial
            == {"id": "a", "name": "alpha", "stock": 3, "reserved": 0}
            and item_a_reserve_result_2
            == {"id": "a", "name": "alpha", "stock": 3, "reserved": 2}
            and item_a_after_reserve_2
            == {"id": "a", "name": "alpha", "stock": 3, "reserved": 2}
            and item_a_reserve_result_3
            == {"id": "a", "name": "alpha", "stock": 3, "reserved": 3}
            and item_a_after_reserve_3
            == {"id": "a", "name": "alpha", "stock": 3, "reserved": 3}
            and lines[9].strip() == "400"
            and lines[10].strip() == "400"
            and json.loads(lines[11].strip())
            == {"id": "a", "name": "alpha", "stock": 3, "reserved": 1}
            and item_a_after_release
            == {"id": "a", "name": "alpha", "stock": 3, "reserved": 1}
            and item_c_after_reserve
            == {"id": "c", "name": "gamma", "stock": 1, "reserved": 1}
            and lines[14].strip() == "400"
            and item_c_after_release
            == {"id": "c", "name": "gamma", "stock": 1, "reserved": 0}
            and all_items
            == [
                {"id": "a", "name": "alpha", "stock": 3, "reserved": 1},
                {"id": "b", "name": "beta", "stock": 5, "reserved": 0},
                {"id": "c", "name": "gamma", "stock": 1, "reserved": 0},
            ]
            and lines[17].strip() == "404"
            and lines[18].strip() == "400"
            and lines[19].strip() == "400"
            and lines[20].strip() == "400"
            and lines[21].strip() == "400"
        )
    except:
        return False


def _check_rate_limiter(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"class RateLimiter\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
limiter = RateLimiter(2, 10)
print(limiter.allow("a", 1))
print(limiter.allow("a", 2))
print(limiter.allow("a", 3))
print(limiter.allow("a", 11))
print(limiter.allow("a", 12))
print(limiter.allow("a", 13))
print(limiter.allow("a", 21))
print(limiter.allow("b", 3))
print(limiter.allow("b", 5))
print(limiter.allow("b", 12))
print(limiter.allow("b", 13))
print(limiter.allow("c", 100))
print(limiter.allow("c", 109))
print(limiter.allow("c", 110))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            "True",
            "True",
            "False",
            "True",
            "True",
            "False",
            "True",
            "True",
            "True",
            "False",
            "True",
            "True",
            "True",
            "True",
        ]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_query_filter_parser(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def evaluate_query\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
row = {"status": "open", "priority": "high", "owner": "alice"}
print(evaluate_query(row, "status == open"))
print(evaluate_query(row, "status == closed OR priority == high"))
print(evaluate_query(row, "status == closed AND priority == high"))
print(evaluate_query(row, "status == open OR priority == low AND owner == bob"))
print(evaluate_query(row, "(status == closed OR priority == low) AND owner == alice"))
print(evaluate_query(row, "status != closed AND (priority == high OR owner == bob)"))
print(evaluate_query(row, "owner == alice AND status != open"))
print(evaluate_query(row, "(status == open AND priority == low) OR owner == alice"))
print(evaluate_query(row, "status == open AND (priority == low OR owner == alice)"))
print(evaluate_query(row, "((status == open))"))
print(evaluate_query(row, "status != open OR priority != high OR owner != alice"))
print(evaluate_query(row, "status == closed OR priority == low OR owner == bob"))
print(evaluate_query(row, "  status   ==   open   AND   owner   ==   alice  "))
print(evaluate_query(row, "  status   ==   open   AND   (priority == high OR owner == bob)  "))
print(evaluate_query(row, "status != open OR (priority == low AND owner == alice)"))
print(evaluate_query(row, "((status == open)) AND owner == alice"))
print(evaluate_query(row, "(status == closed OR priority == low) AND owner == alice"))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = [
            "True",
            "True",
            "False",
            "True",
            "False",
            "True",
            "False",
            "True",
            "True",
            "True",
            "False",
            "False",
            "True",
            "True",
            "False",
            "True",
            "False",
        ]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_process_logs(response: str) -> bool:
    code = _extract_code(response)
    import ast
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def process_logs\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
logs1 = [
    {"id":"a","timestamp":2,"value":10},
    {"id":"b","timestamp":1,"value":5},
    {"id":"a","timestamp":3,"value":20},
]
logs2 = [
    {"id":"x","timestamp":5,"value":7},
    {"id":"y","timestamp":5,"value":2},
    {"id":"z","timestamp":6,"value":4},
    {"id":"y","timestamp":1,"value":100},
]
logs3 = [
    {"id":"n1","timestamp":9,"value":10},
    {"id":"n2","timestamp":9,"value":4},
    {"id":"n3","timestamp":8,"value":6},
    {"id":"n4","timestamp":9,"value":9},
    {"id":"n2","timestamp":0,"value":999},
]
logs4 = [
    {"id":"s1","timestamp":1,"value":5},
    {"id":"s2","timestamp":1,"value":7},
    {"id":"s3","timestamp":2,"value":9},
]
logs5 = [
    {"id":"d","timestamp":3,"value":10},
    {"id":"e","timestamp":2,"value":4},
    {"id":"d","timestamp":1,"value":99},
    {"id":"f","timestamp":2,"value":6},
]
logs6 = [
    {"id":"u1","timestamp":4,"value":8},
    {"id":"u2","timestamp":4,"value":3},
    {"id":"u1","timestamp":7,"value":11},
    {"id":"u3","timestamp":7,"value":1},
    {"id":"u4","timestamp":12,"value":5},
    {"id":"u2","timestamp":14,"value":9},
]
print(process_logs(logs1))
print(process_logs(logs2))
print(process_logs(logs3))
print(process_logs([]))
print(process_logs(logs4))
print(process_logs(logs5))
print(process_logs(logs6))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected1 = [
            {"id": "b", "timestamp": 1, "value": 5, "delta": 0},
            {"id": "a", "timestamp": 2, "value": 10, "delta": 5},
        ]
        expected2 = [
            {"id": "x", "timestamp": 5, "value": 7, "delta": 0},
            {"id": "y", "timestamp": 5, "value": 2, "delta": -5},
            {"id": "z", "timestamp": 6, "value": 4, "delta": 2},
        ]
        expected3 = [
            {"id": "n3", "timestamp": 8, "value": 6, "delta": 0},
            {"id": "n1", "timestamp": 9, "value": 10, "delta": 4},
            {"id": "n2", "timestamp": 9, "value": 4, "delta": -6},
            {"id": "n4", "timestamp": 9, "value": 9, "delta": 5},
        ]
        expected4 = [
            {"id": "s1", "timestamp": 1, "value": 5, "delta": 0},
            {"id": "s2", "timestamp": 1, "value": 7, "delta": 2},
            {"id": "s3", "timestamp": 2, "value": 9, "delta": 2},
        ]
        expected5 = [
            {"id": "e", "timestamp": 2, "value": 4, "delta": 0},
            {"id": "f", "timestamp": 2, "value": 6, "delta": 2},
            {"id": "d", "timestamp": 3, "value": 10, "delta": 4},
        ]
        expected6 = [
            {"id": "u1", "timestamp": 4, "value": 8, "delta": 0},
            {"id": "u2", "timestamp": 4, "value": 3, "delta": -5},
            {"id": "u3", "timestamp": 7, "value": 1, "delta": -2},
            {"id": "u4", "timestamp": 12, "value": 5, "delta": 4},
        ]
        return (
            len(lines) == 7
            and ast.literal_eval(lines[0].strip()) == expected1
            and ast.literal_eval(lines[1].strip()) == expected2
            and ast.literal_eval(lines[2].strip()) == expected3
            and ast.literal_eval(lines[3].strip()) == []
            and ast.literal_eval(lines[4].strip()) == expected4
            and ast.literal_eval(lines[5].strip()) == expected5
            and ast.literal_eval(lines[6].strip()) == expected6
        )
    except:
        return False


def _check_shortest_path_with_break(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def shortest_path_with_break\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
cases = [
    ([[0]], 1),
    ([[0,1],[0,0]], 3),
    ([[0,1,1],[1,1,0],[1,1,0]], -1),
    ([[0,1,0],[0,1,0],[0,0,0]], 5),
    ([[0,1,0],[1,1,0],[0,0,0]], 5),
    ([[0,0,0],[1,1,0],[0,0,0]], 5),
    ([[0,1,1,0],[0,0,1,0],[1,0,0,0]], 6),
    ([[0,1,1,1],[1,1,1,0],[0,0,0,0],[0,1,1,0]], 7),
]
for grid, _ in cases:
    print(shortest_path_with_break(grid))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = ["1", "3", "-1", "5", "5", "5", "6", "7"]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_job_queue_service(response: str) -> bool:
    code = _extract_code(response)
    import json
    import re
    import tempfile
    from pathlib import Path

    if "FastAPI" not in code or not re.search(r"\bapp\s*=\s*FastAPI\(", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
import inspect
import json
from fastapi import HTTPException

def to_jsonable(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict") and callable(value.dict):
        value = value.dict()

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value

def find_route(path, method):
    for route in app.routes:
        methods = getattr(route, "methods", set())
        if getattr(route, "path", None) == path and method in methods:
            return route
    raise AssertionError(f"missing route {method} {path}")

def make_body(route, **kwargs):
    params = list(inspect.signature(route.endpoint).parameters.values())
    assert params, "route must accept a body parameter"
    annotation = params[-1].annotation
    return annotation(**kwargs)

def expect_status(status_code, fn, *args):
    try:
        fn(*args)
    except HTTPException as exc:
        print(exc.status_code)
        return
    raise AssertionError(f"expected HTTPException {status_code}")

post_jobs = find_route("/jobs", "POST")
get_job = find_route("/jobs/{id}", "GET") if any(getattr(route, "path", None) == "/jobs/{id}" for route in app.routes) else find_route("/jobs/{job_id}", "GET")
start_job = find_route("/jobs/{id}/start", "POST") if any(getattr(route, "path", None) == "/jobs/{id}/start" for route in app.routes) else find_route("/jobs/{job_id}/start", "POST")
finish_job = find_route("/jobs/{id}/finish", "POST") if any(getattr(route, "path", None) == "/jobs/{id}/finish" for route in app.routes) else find_route("/jobs/{job_id}/finish", "POST")
list_jobs = find_route("/jobs", "GET")

print(json.dumps(to_jsonable(post_jobs.endpoint(make_body(post_jobs, id="b", payload="beta"))), sort_keys=True))
print(json.dumps(to_jsonable(post_jobs.endpoint(make_body(post_jobs, id="a", payload="alpha"))), sort_keys=True))
print(json.dumps(to_jsonable(post_jobs.endpoint(make_body(post_jobs, id="c", payload="gamma"))), sort_keys=True))
expect_status(400, post_jobs.endpoint, make_body(post_jobs, id="a", payload="duplicate"))
expect_status(400, finish_job.endpoint, "a")
print(json.dumps(to_jsonable(start_job.endpoint("a")), sort_keys=True))
expect_status(400, start_job.endpoint, "a")
print(json.dumps(to_jsonable(finish_job.endpoint("a")), sort_keys=True))
expect_status(400, finish_job.endpoint, "a")
print(json.dumps(to_jsonable(get_job.endpoint("a")), sort_keys=True))
print(json.dumps(to_jsonable(list_jobs.endpoint()), sort_keys=True))
expect_status(404, get_job.endpoint, "missing")
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=20
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        if len(lines) != 12:
            return False
        return (
            json.loads(lines[0].strip())
            == {"id": "b", "status": "queued", "payload": "beta"}
            and json.loads(lines[1].strip())
            == {"id": "a", "status": "queued", "payload": "alpha"}
            and json.loads(lines[2].strip())
            == {"id": "c", "status": "queued", "payload": "gamma"}
            and lines[3].strip() == "400"
            and lines[4].strip() == "400"
            and json.loads(lines[5].strip())
            == {"id": "a", "status": "running", "payload": "alpha"}
            and lines[6].strip() == "400"
            and json.loads(lines[7].strip())
            == {"id": "a", "status": "done", "payload": "alpha"}
            and lines[8].strip() == "400"
            and json.loads(lines[9].strip())
            == {"id": "a", "status": "done", "payload": "alpha"}
            and json.loads(lines[10].strip())
            == [
                {"id": "a", "status": "done", "payload": "alpha"},
                {"id": "b", "status": "queued", "payload": "beta"},
                {"id": "c", "status": "queued", "payload": "gamma"},
            ]
            and lines[11].strip() == "404"
        )
    except:
        return False


def _check_arithmetic_evaluator(response: str) -> bool:
    code = _extract_code(response)
    import re
    import tempfile
    from pathlib import Path

    if not re.search(r"def evaluate\b", code):
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                code
                + """
print(evaluate("-3 + 5"))
print(evaluate("2*(-3 + 4)"))
print(evaluate("18 / (3 + 1)"))
print(evaluate("7 + -(2*3)"))
print(evaluate("-(1 + 2) * 3"))
print(evaluate("14 / 3"))
print(evaluate("14 / -3"))
print(evaluate("-14 / 3"))
print(evaluate("(-2)*(-3) + 4"))
print(evaluate("1 - -2 * (3 + 4)"))
"""
            )
            t = f.name
        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
        )
        Path(t).unlink(missing_ok=True)
        if result.returncode != 0:
            return False
        lines = result.stdout.splitlines()
        expected = ["2", "2", "4", "1", "-9", "4", "-4", "-4", "10", "15"]
        return len(lines) == len(expected) and all(
            lines[i].strip() == expected[i] for i in range(len(expected))
        )
    except:
        return False


def _check_debug_test(code: str, test_name: str) -> tuple[int, int]:
    """Check debugging test correctness - run with test cases.

    Returns: (num_correct, total_tests)
    """
    import ast
    import re
    import tempfile
    from pathlib import Path

    test_cases = {
        "binary_search": [
            ([4, 5, 6, 7, 0, 1, 2], 0, 4),
            ([4, 5, 6, 7, 0, 1, 2], 3, -1),
            ([1], 1, 0),
            ([1], 0, -1),
            ([1, 3], 3, 1),
            ([3, 1], 1, 1),
            ([5, 1, 3], 3, 2),
            ([], 9, -1),
            ([8, 9, 2, 3, 4], 8, 0),
            ([6, 7, 1, 2, 3, 4, 5], 4, 5),
            ([30, 40, 50, 10, 20], 20, 4),
            ([10, 20, 30, 40, 50], 40, 3),
            ([11, 13, 15, 17], 13, 1),
            ([15, 18, 2, 3, 6, 12], 12, 5),
        ],
        "quicksort": [
            ([3, 1, 2], [1, 2, 3]),
            ([5, 3, 8, 1, 2], [1, 2, 3, 5, 8]),
            ([1], [1]),
            ([2, 2, 2], [2, 2, 2]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([1, 2], [1, 2]),
            ([2, 1], [1, 2]),
            ([1, 3, 2], [1, 2, 3]),
            ([4, 2, 5, 3, 1], [1, 2, 3, 4, 5]),
            ([6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]),
        ],
        "duplicates": [
            ([1, 1, 2, 2, 3, 3], [1, 2, 3]),
            ([1, 2, 3], []),
            ([1, 1, 1], [1, 1]),
            ([4, 4, 5, 5, 6], [4, 5]),
            ([], []),
            ([7], []),
            ([1, 2, 2, 3, 3, 3], [2, 3, 3]),
            ([5, 5], [5]),
            ([8, 8, 8, 9], [8, 8]),
            ([10, 11, 11, 12], [11]),
            ([2, 2, 2, 2], [2, 2, 2]),
            ([3, 1, 3, 1, 3], [3, 1, 3]),
        ],
        "two_sum": [
            ([2, 7, 11, 15], 9, [0, 1]),
            ([3, 2, 4], 6, [1, 2]),
            ([3, 3], 6, [0, 1]),
            ([1, 2, 3, 4, 5], 9, [3, 4]),
            ([-1, -2, -3, -4, -5], -6, [1, 3]),
            ([0, 4, 3, 0], 0, [0, 3]),
            ([2, 5, 7], 9, [0, 2]),
            ([1, 1], 2, [0, 1]),
            ([-1, 0], -1, [0, 1]),
            ([1, 2, 3], 5, [1, 2]),
            ([10, -2, 4, -8, 6], 2, [1, 4]),
            ([5, 5, 5], 10, [0, 1]),
        ],
        "merge": [
            ([1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6]),
            ([], [1, 2], [1, 2]),
            ([1, 2], [], [1, 2]),
            ([1], [2], [1, 2]),
            ([2], [1], [1, 2]),
            ([1, 3, 5, 7], [2, 4, 6, 8], [1, 2, 3, 4, 5, 6, 7, 8]),
            ([1, 1, 1], [1, 1, 1], [1, 1, 1, 1, 1, 1]),
            ([], [], []),
            ([5], [1, 2, 3, 4], [1, 2, 3, 4, 5]),
            ([1, 3, 5], [2, 4], [1, 2, 3, 4, 5]),
            ([1, 2, 2, 9], [2, 2, 3], [1, 2, 2, 2, 2, 3, 9]),
            ([-5, -1], [-3, -2, 4], [-5, -3, -2, -1, 4]),
        ],
        "first_missing_positive": [
            ([1, 2, 0], 3),
            ([3, 4, -1, 1], 2),
            ([7, 8, 9, 11, 12], 1),
            ([1, 2, 3], 4),
            ([2], 1),
            ([], 1),
            ([1], 2),
            ([2, 2], 1),
            ([1, 1, 0, -1, -2], 2),
            ([4, 3, 2, 1], 5),
            ([1, 2, 6, 3, 5, 4], 7),
            ([0, -10, 1, 3, -20], 2),
        ],
        "merge_intervals": [
            ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
            ([[1, 4], [4, 5]], [[1, 5]]),
            ([[1, 4], [2, 3]], [[1, 4]]),
            ([[5, 7], [1, 2], [2, 4]], [[1, 4], [5, 7]]),
            ([], []),
            ([[1, 4], [0, 2], [3, 5]], [[0, 5]]),
            ([[1, 2], [3, 5]], [[1, 2], [3, 5]]),
            ([[1, 5], [2, 3]], [[1, 5]]),
            ([[1, 4], [0, 1]], [[0, 4]]),
            ([[2, 3], [4, 6], [6, 7]], [[2, 3], [4, 7]]),
            ([[5, 1], [2, 2], [2, 8]], [[1, 8]]),
            ([[0, 0], [1, 1]], [[0, 0], [1, 1]]),
        ],
        "decode_rle": [
            ("3a2b1c", "aaabbc"),
            ("10x", "xxxxxxxxxx"),
            ("2a12b", "aabbbbbbbbbbbb"),
            ("", ""),
            ("1z1y1x", "zyx"),
            ("01a", "a"),
            ("3a0b", "aaa"),
            ("12a", "aaaaaaaaaaaa"),
            ("2a3b4c", "aabbbcccc"),
            ("2a10b1c", "aabbbbbbbbbbc"),
            ("a", ValueError),
            ("2", ValueError),
            ("0003a", "aaa"),
            ("1a01b", "ab"),
        ],
        "first_occurrence": [
            ([1, 2, 2, 2, 3], 2, 1),
            ([1, 1, 1, 1], 1, 0),
            ([1, 2, 3, 4], 3, 2),
            ([1], 1, 0),
            ([1], 2, -1),
            ([], 1, -1),
            ([1, 3, 5, 7], 6, -1),
            ([2, 2, 2, 3, 4], 2, 0),
            ([1, 2, 3, 3, 3, 4], 3, 2),
            ([1, 2, 4, 4, 4, 5], 4, 2),
            ([0, 1, 1, 1, 2], 0, 0),
            ([5, 5, 5, 5, 5], 5, 0),
        ],
        "cycle": [
            (None, False),
        ],
        "session_aggregator": [
            (
                [("a", "e1", 1), ("a", "e2", 5), ("a", "e3", 15)],
                {"a": [["e1", "e2", "e3"]]},
            ),
            (
                [("a", "e1", 1), ("b", "e2", 2), ("a", "e3", 11), ("b", "e4", 20)],
                {"a": [["e1", "e3"]], "b": [["e2"], ["e4"]]},
            ),
            (
                [("u", "x1", 10), ("u", "x2", 21), ("u", "x3", 31)],
                {"u": [["x1"], ["x2", "x3"]]},
            ),
            (
                [],
                {},
            ),
            (
                [("a", "e1", 1), ("a", "e2", 12), ("a", "e3", 22), ("a", "e4", 40)],
                {"a": [["e1"], ["e2", "e3"], ["e4"]]},
            ),
            (
                [("a", "e1", 1), ("a", "e2", 11), ("a", "e3", 22)],
                {"a": [["e1", "e2"], ["e3"]]},
            ),
            (
                [
                    ("a", "e1", 1),
                    ("b", "e2", 1),
                    ("a", "e3", 20),
                    ("b", "e4", 12),
                    ("b", "e5", 23),
                ],
                {"a": [["e1"], ["e3"]], "b": [["e2"], ["e4"], ["e5"]]},
            ),
            (
                [("solo", "only", 99)],
                {"solo": [["only"]]},
            ),
        ],
        "sliding_window_maximum": [
            ([1, 3, -1, -3, 5, 3, 6, 7], 3, [3, 3, 5, 5, 6, 7]),
            ([1], 1, [1]),
            ([], 1, []),
            ([4, 2], 1, [4, 2]),
            ([9, 11], 2, [11]),
            ([7, 2, 4], 2, [7, 4]),
            ([1, 1, 1, 1], 2, [1, 1, 1]),
            ([9, 10, 9, -7, -4, -8, 2, -6], 5, [10, 10, 9, 2]),
        ],
        "rotate_matrix": [
            ([[1, 2], [3, 4]], [[3, 1], [4, 2]]),
            (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[7, 4, 1], [8, 5, 2], [9, 6, 3]],
            ),
            ([[1]], [[1]]),
        ],
        "product_except_self": [
            ([1, 2, 3, 4], [24, 12, 8, 6]),
            ([0, 1, 2, 3], [6, 0, 0, 0]),
            ([0, 0, 2], [0, 0, 0]),
            ([5], [1]),
            ([-1, 1, -1, 1], [-1, 1, -1, 1]),
            ([1, 2, 0, 4], [0, 0, 8, 0]),
        ],
        "kth_smallest_bst": [
            ([3, 1, 4, None, 2], 1, 1),
            ([5, 3, 6, 2, 4, None, None, 1], 3, 3),
            ([1], 1, 1),
            ([2, 1, 3], 2, 2),
            ([5, 3, 7, 2, 4, 6, 8], 5, 6),
        ],
        "coin_change": [
            ([1, 2, 5], 11, 3),
            ([2], 3, -1),
            ([1], 0, 0),
            ([2, 5, 10, 1], 27, 4),
            ([186, 419, 83, 408], 6249, 20),
            ([1, 3, 4], 6, 2),
        ],
        "reorder_logs": [
            (
                [
                    "dig1 8 1 5 1",
                    "let1 art can",
                    "dig2 3 6",
                    "let2 own kit dig",
                    "let3 art zero",
                ],
                [
                    "let1 art can",
                    "let3 art zero",
                    "let2 own kit dig",
                    "dig1 8 1 5 1",
                    "dig2 3 6",
                ],
            ),
            ([], []),
            (
                [
                    "dig1 0 1",
                    "let1 art can",
                    "dig2 1 2",
                    "let2 art zero",
                    "let3 art can",
                ],
                [
                    "let1 art can",
                    "let3 art can",
                    "let2 art zero",
                    "dig1 0 1",
                    "dig2 1 2",
                ],
            ),
        ],
        "interval_insert": [
            ([[1, 3], [6, 9]], [2, 5], [[1, 5], [6, 9]]),
            (
                [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]],
                [4, 8],
                [[1, 2], [3, 10], [12, 16]],
            ),
            ([], [5, 7], [[5, 7]]),
            ([[1, 5]], [2, 3], [[1, 5]]),
            ([[1, 2], [5, 6]], [3, 4], [[1, 2], [3, 4], [5, 6]]),
            ([[1, 2], [3, 5]], [6, 7], [[1, 2], [3, 5], [6, 7]]),
        ],
        "ttl_cache": [
            ("basic-expiry", ["10", "-1", "-1"]),
            ("cleanup-and-survivors", ["-1", "2", "-1", "[]"]),
        ],
        "leaderboard": [
            (
                "rankings",
                [
                    ["carol", "bob"],
                    ["carol", "bob", "dave", "alice"],
                    ["alice", "carol", "bob"],
                    [],
                ],
            ),
        ],
        "median_two_sorted_arrays_debug": [
            ([1, 3], [2], 2.0),
            ([1, 2], [3, 4], 2.5),
            ([], [1], 1.0),
            ([2], [], 2.0),
            ([0, 0], [0, 0], 0.0),
            ([1], [2, 3, 4, 5, 6], 3.5),
            ([-5, -3, -1], [-2, 4, 8, 10], -1.0),
        ],
        "min_window_debug": [
            ("ADOBECODEBANC", "ABC", "BANC"),
            ("a", "a", "a"),
            ("a", "aa", ""),
            ("aa", "aa", "aa"),
            ("bba", "ab", "ba"),
            ("cabwefgewcwaefgcf", "cae", "cwae"),
            ("anything", "", ""),
        ],
        "skyline_problem_debug": [
            ([], []),
            (
                [[2, 9, 10], [3, 7, 15], [5, 12, 12]],
                [[2, 10], [3, 15], [7, 12], [12, 0]],
            ),
            (
                [[1, 3, 4], [3, 6, 4]],
                [[1, 4], [6, 0]],
            ),
            (
                [[1, 5, 3], [1, 5, 4], [1, 5, 2]],
                [[1, 4], [5, 0]],
            ),
            (
                [[0, 2, 3], [2, 5, 3]],
                [[0, 3], [5, 0]],
            ),
            (
                [
                    [1, 5, 11],
                    [2, 7, 6],
                    [3, 9, 13],
                    [12, 16, 7],
                    [14, 25, 3],
                    [19, 22, 18],
                    [23, 29, 13],
                    [24, 28, 4],
                ],
                [
                    [1, 11],
                    [3, 13],
                    [9, 0],
                    [12, 7],
                    [16, 3],
                    [19, 18],
                    [22, 3],
                    [23, 13],
                    [29, 0],
                ],
            ),
            (
                [[1, 2, 1], [1, 2, 2], [1, 2, 3]],
                [[1, 3], [2, 0]],
            ),
        ],
        "word_search_ii_debug": [
            (
                [
                    ["o", "a", "a", "n"],
                    ["e", "t", "a", "e"],
                    ["i", "h", "k", "r"],
                    ["i", "f", "l", "v"],
                ],
                ["oath", "pea", "eat", "rain"],
                ["eat", "oath"],
            ),
            (
                [
                    ["o", "a", "a", "n"],
                    ["e", "t", "a", "e"],
                    ["i", "h", "k", "r"],
                    ["i", "f", "l", "v"],
                ],
                ["oat", "oath", "oaths", "eat", "rain", "oath"],
                ["eat", "oat", "oath"],
            ),
            (
                [],
                ["a"],
                [],
            ),
            (
                [["a", "a"], ["a", "a"]],
                ["a", "aa", "aaa", "aaaa", "aaaaa"],
                ["a", "aa", "aaa", "aaaa"],
            ),
            (
                [["a", "b"], ["c", "d"]],
                ["ab", "ac", "bd", "ca", "abdc", "dc"],
                ["ab", "abdc", "ac", "bd", "ca", "dc"],
            ),
            (
                [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]],
                ["abe", "cfi", "adg", "beh", "defi", "aei"],
                ["abe", "adg", "beh", "cfi", "defi"],
            ),
        ],
        "alien_dictionary_debug": [
            (["wrt", "wrf", "er", "ett", "rftt"], True),
            (["z", "x"], True),
            (["z", "x", "z"], False),
            (["abc", "ab"], False),
            (["baa", "abcd", "abca", "cab", "cad"], True),
        ],
        "candy_crush_debug": [
            (
                [
                    [110, 5, 112, 113, 114],
                    [210, 211, 5, 213, 214],
                    [310, 311, 3, 313, 314],
                    [410, 411, 412, 5, 414],
                    [5, 1, 512, 3, 3],
                    [610, 4, 1, 613, 614],
                    [710, 1, 2, 713, 714],
                    [810, 1, 2, 1, 1],
                    [1, 1, 2, 2, 2],
                    [4, 1, 4, 4, 1014],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [110, 0, 0, 0, 114],
                    [210, 0, 0, 0, 214],
                    [310, 0, 0, 113, 314],
                    [410, 0, 0, 213, 414],
                    [610, 211, 112, 313, 614],
                    [710, 311, 412, 613, 714],
                    [810, 411, 512, 713, 1014],
                ],
            ),
            (
                [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ),
            (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ),
            (
                [[1, 1, 1, 2], [3, 4, 5, 6], [7, 8, 9, 9], [7, 7, 7, 9]],
                [[0, 0, 0, 2], [0, 0, 0, 6], [3, 4, 5, 9], [7, 8, 9, 9]],
            ),
            (
                [[1, 2, 3, 4], [1, 5, 6, 4], [1, 7, 8, 4], [9, 9, 9, 4]],
                [[0, 0, 0, 0], [0, 2, 3, 0], [0, 5, 6, 0], [0, 7, 8, 0]],
            ),
        ],
        "substring_concatenation_debug": [
            ("barfoothefoobarman", ["foo", "bar"], [0, 9]),
            ("wordgoodgoodgoodbestword", ["word", "good", "best", "word"], []),
            ("barfoofoobarthefoobarman", ["bar", "foo", "the"], [6, 9, 12]),
            (
                "lingmindraboofooowingdingbarrwingmonkeypoundcake",
                ["fooo", "barr", "wing", "ding", "wing"],
                [13],
            ),
            ("", ["foo"], []),
            ("aaaaaa", ["aa", "aa"], [0, 1, 2]),
        ],
    }

    if test_name not in test_cases:
        return 0, 0

    tests = test_cases[test_name]
    call_names = {
        "quicksort": "quick_sort",
        "duplicates": "find_duplicates",
        "merge": "merge_sorted",
        "session_aggregator": "group_sessions",
        "sliding_window_maximum": "max_sliding_window",
        "rotate_matrix": "rotate",
        "product_except_self": "product_except_self",
        "kth_smallest_bst": "kth_smallest",
        "coin_change": "coin_change",
        "reorder_logs": "reorder_logs",
        "interval_insert": "insert_interval",
        "median_two_sorted_arrays_debug": "find_median_sorted_arrays",
        "min_window_debug": "min_window",
        "skyline_problem_debug": "get_skyline",
        "word_search_ii_debug": "find_words",
        "alien_dictionary_debug": "alien_order",
        "candy_crush_debug": "candy_crush",
        "substring_concatenation_debug": "find_substring",
    }
    call_name = call_names.get(test_name, test_name)

    def build_script(body: str, calls: list[str]) -> str:
        parts = [body.rstrip("\n")]
        parts.extend(call.rstrip("\n") for call in calls)
        return "\n".join(parts) + "\n"

    # For cycle we can't easily test, just check it runs
    if test_name == "cycle":
        if not re.search(r"def has_cycle", code):
            return 0, len(tests)
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    code
                    + """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build(values, cycle_at=-1):
    if not values:
        return None
    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if 0 <= cycle_at < len(nodes):
        nodes[-1].next = nodes[cycle_at]
    return nodes[0]

print(has_cycle(build([])))
print(has_cycle(build([1])))
print(has_cycle(build([1], 0)))
print(has_cycle(build([1,2,3,4], 1)))
print(has_cycle(build([1,2,3,4], -1)))
print(has_cycle(build([7,7,7,7], -1)))
print(has_cycle(build([9,8,7,6,5], 3)))
"""
                )
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, 7
            lines = result.stdout.splitlines()
            expected = ["False", "False", "True", "True", "False", "False", "True"]
            num_correct = 0
            for i, expected_value in enumerate(expected):
                if i < len(lines) and lines[i].strip() == expected_value:
                    num_correct += 1
            return num_correct, len(expected)
        except:
            return 0, 7

    if test_name == "ttl_cache":
        if not re.search(r"class TTLCache\b", code):
            return 0, 7
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    code
                    + """
c = TTLCache()
c.put("a", 10, ttl=5, now=0)
print(c.get("a", 4))
print(c.get("a", 5))
print(c.get("a", 6))

c.put("x", 1, ttl=1, now=7)
c.put("y", 2, ttl=3, now=7)
c.cleanup(8)
print(c.get("x", 8))
print(c.get("y", 8))
c.cleanup(10)
print(c.get("y", 10))
print(sorted(c.data.keys()))
"""
                )
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, 7
            lines = result.stdout.splitlines()
            expected = ["10", "-1", "-1", "-1", "2", "-1", "[]"]
            num_correct = 0
            for i, expected_value in enumerate(expected):
                if i < len(lines) and lines[i].strip() == expected_value:
                    num_correct += 1
            return num_correct, len(expected)
        except:
            return 0, 7

    if test_name == "leaderboard":
        if not re.search(r"class Leaderboard\b", code):
            return 0, 4
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    code
                    + """
lb = Leaderboard()
lb.add_score("bob", 5)
lb.add_score("alice", 5)
lb.add_score("bob", 2)
lb.add_score("carol", 9)
lb.add_score("dave", 7)
print(lb.top(2))
print(lb.top(10))
lb.add_score("alice", 4)
print(lb.top(3))
print(lb.top(0))
lb.add_score("erin", 7)
print(lb.top(4))
"""
                )
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, 5
            lines = result.stdout.splitlines()
            expected = [
                ["carol", "bob"],
                ["carol", "bob", "dave", "alice"],
                ["alice", "carol", "bob"],
                [],
                ["alice", "carol", "bob", "dave"],
            ]
            num_correct = 0
            for i, expected_value in enumerate(expected):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == expected_value:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(expected)
        except:
            return 0, 5

    if test_name == "median_two_sorted_arrays_debug":
        if not re.search(r"def find_median_sorted_arrays\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [
                    f"print(find_median_sorted_arrays({nums1}, {nums2}))"
                    for nums1, nums2, _ in tests
                ]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, _, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if abs(float(lines[i].strip()) - expected) < 1e-9:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "min_window_debug":
        if not re.search(r"def min_window\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [f"print(min_window({s!r}, {tval!r}))" for s, tval, _ in tests]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, _, expected) in enumerate(tests):
                if i < len(lines) and lines[i].strip() == expected:
                    num_correct += 1
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "skyline_problem_debug":
        if not re.search(r"def get_skyline\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [f"print(get_skyline({buildings}))" for buildings, _ in tests]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == expected:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "word_search_ii_debug":
        if not re.search(r"def find_words\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [f"print(find_words({board}, {words}))" for board, words, _ in tests]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, _, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        got = sorted(set(ast.literal_eval(lines[i].strip())))
                        if got == sorted(expected):
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "alien_dictionary_debug":
        if not re.search(r"def alien_order\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    code
                    + """
def is_valid_order(words, order):
    chars = set(''.join(words))
    if set(order) != chars or len(order) != len(chars):
        return False
    pos = {ch: i for i, ch in enumerate(order)}
    for w1, w2 in zip(words, words[1:]):
        if len(w1) > len(w2) and w1.startswith(w2):
            return False
        for a, b in zip(w1, w2):
            if a != b:
                if pos[a] > pos[b]:
                    return False
                break
    return True

cases = [
    (["wrt", "wrf", "er", "ett", "rftt"], True),
    (["z", "x"], True),
    (["z", "x", "z"], False),
    (["abc", "ab"], False),
    (["baa", "abcd", "abca", "cab", "cad"], True),
]
for words, expect_valid in cases:
    order = alien_order(words)
    print(is_valid_order(words, order) if expect_valid else order == '')
"""
                )
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            expected = ["True"] * len(tests)
            if len(lines) != len(expected):
                return 0, len(tests)
            if all(lines[i].strip() == expected[i] for i in range(len(expected))):
                return len(tests), len(tests)
            return 0, len(tests)
        except:
            return 0, len(tests)

    if test_name == "candy_crush_debug":
        if not re.search(r"def candy_crush\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    code
                    + """
def run(board):
    b = [row[:] for row in board]
    out = candy_crush(b)
    print(out if out is not None else b)

cases = [
    [[110,5,112,113,114],[210,211,5,213,214],[310,311,3,313,314],[410,411,412,5,414],[5,1,512,3,3],[610,4,1,613,614],[710,1,2,713,714],[810,1,2,1,1],[1,1,2,2,2],[4,1,4,4,1014]],
    [[1,1,1],[2,2,2],[3,3,3]],
    [[1,2,3],[4,5,6],[7,8,9]],
    [[1,1,1,2],[3,4,5,6],[7,8,9,9],[7,7,7,9]],
    [[1,2,3,4],[1,5,6,4],[1,7,8,4],[9,9,9,4]],
]
for board in cases:
    run(board)
"""
                )
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=20
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == expected:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "substring_concatenation_debug":
        if not re.search(r"def find_substring\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [f"print(find_substring({s!r}, {words}))" for s, words, _ in tests]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, _, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == expected:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "sliding_window_maximum":
        if not re.search(r"def max_sliding_window\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [f"print(max_sliding_window({nums}, {k}))" for nums, k, _ in tests]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, _, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == expected:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "rotate_matrix":
        if not re.search(r"def rotate\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    code
                    + """
import copy
cases = [
    [[1, 2], [3, 4]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[1]],
]
for matrix in cases:
    original = copy.deepcopy(matrix)
    rotate(matrix)
    print(matrix)
"""
                )
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            expected = [tc[1] for tc in tests]
            num_correct = 0
            for i, exp in enumerate(expected):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == exp:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "product_except_self":
        if not re.search(r"def product_except_self\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [f"print(product_except_self({nums}))" for nums, _ in tests]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == expected:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "kth_smallest_bst":
        if not re.search(r"def kth_smallest\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    code
                    + """
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(values):
    if not values:
        return None
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[::-1]
    root = kids.pop()
    for node in nodes:
        if node is not None:
            if kids:
                node.left = kids.pop()
            if kids:
                node.right = kids.pop()
    return root

cases = [
    ([3, 1, 4, None, 2], 1),
    ([5, 3, 6, 2, 4, None, None, 1], 3),
    ([1], 1),
    ([2, 1, 3], 2),
    ([5, 3, 7, 2, 4, 6, 8], 5),
]
for values, k in cases:
    root = build_tree(values)
    print(kth_smallest(root, k))
"""
                )
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            expected = [tc[2] for tc in tests]
            num_correct = 0
            for i, exp in enumerate(expected):
                if i < len(lines):
                    try:
                        if int(lines[i].strip()) == exp:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "coin_change":
        if not re.search(r"def coin_change\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [f"print(coin_change({coins}, {amount}))" for coins, amount, _ in tests]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, _, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if int(lines[i].strip()) == expected:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "reorder_logs":
        if not re.search(r"def reorder_logs\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [f"print(reorder_logs({logs}))" for logs, _ in tests]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == expected:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    if test_name == "interval_insert":
        if not re.search(r"def insert_interval\b", code):
            return 0, len(tests)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                calls = [
                    f"print(insert_interval({intervals}, {new_interval}))"
                    for intervals, new_interval, _ in tests
                ]
                f.write(build_script(code, calls))
                t = f.name
            result = subprocess.run(
                [PYTHON_BIN, t], capture_output=True, text=True, timeout=10
            )
            Path(t).unlink(missing_ok=True)
            if result.returncode != 0:
                return 0, len(tests)
            lines = result.stdout.splitlines()
            num_correct = 0
            for i, (_, _, expected) in enumerate(tests):
                if i < len(lines):
                    try:
                        if ast.literal_eval(lines[i].strip()) == expected:
                            num_correct += 1
                    except:
                        pass
            return num_correct, len(tests)
        except:
            return 0, len(tests)

    function_names = {
        "quicksort": ["quick_sort", "quicksort"],
        "duplicates": ["find_duplicates", "duplicates"],
        "merge": ["merge_sorted", "merge"],
        "session_aggregator": ["group_sessions"],
        "sliding_window_maximum": ["max_sliding_window"],
        "rotate_matrix": ["rotate"],
        "product_except_self": ["product_except_self"],
        "kth_smallest_bst": ["kth_smallest"],
        "coin_change": ["coin_change"],
        "reorder_logs": ["reorder_logs"],
        "interval_insert": ["insert_interval"],
    }
    candidate_names = function_names.get(test_name, [test_name])
    if not any(
        re.search(rf"def {re.escape(name)}\b", code) for name in candidate_names
    ):
        return 0, len(tests)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            calls = []
            for tc in tests:
                if test_name == "binary_search":
                    arr, target, _ = tc
                    calls.append(f"print({call_name}({arr}, {target}))")
                elif test_name in ["quicksort", "duplicates", "session_aggregator"]:
                    arr, _ = tc
                    calls.append(f"print({call_name}({arr}))")
                elif test_name == "two_sum":
                    arr, target, _ = tc
                    calls.append(f"print({call_name}({arr}, {target}))")
                elif test_name == "merge":
                    arr1, arr2, _ = tc
                    calls.append(f"print({call_name}({arr1}, {arr2}))")
                elif test_name in [
                    "first_missing_positive",
                    "merge_intervals",
                    "decode_rle",
                    "first_occurrence",
                ]:
                    if test_name == "first_missing_positive":
                        arr, _ = tc
                        calls.append(f"print({call_name}({arr}))")
                    elif test_name == "merge_intervals":
                        intervals, _ = tc
                        calls.append(f"print({call_name}({intervals}))")
                    elif test_name == "decode_rle":
                        s, expected = tc
                        if expected is ValueError:
                            calls.append(
                                f"try:\n    print({call_name}({s!r}))\nexcept Exception as e:\n    print(f'EXC:{{type(e).__name__}}')"
                            )
                        else:
                            calls.append(f"print({call_name}({s!r}))")
                    elif test_name == "first_occurrence":
                        arr, target, _ = tc
                        calls.append(f"print({call_name}({arr}, {target}))")
            f.write(build_script(code, calls))
            t = f.name

        result = subprocess.run(
            [PYTHON_BIN, t], capture_output=True, text=True, timeout=15
        )
        Path(t).unlink(missing_ok=True)

        if result.returncode != 0:
            return 0, len(tests)

        lines = result.stdout.strip().split("\n")

        num_correct = 0
        for i, tc in enumerate(tests):
            if i >= len(lines):
                break
            actual = lines[i].strip()
            try:
                if test_name == "binary_search":
                    expected = tc[2]
                    if int(actual) == expected:
                        num_correct += 1
                elif test_name in ["quicksort", "duplicates", "session_aggregator"]:
                    expected = tc[1]
                    parsed = ast.literal_eval(actual)
                    if parsed == expected:
                        num_correct += 1
                elif test_name == "two_sum":
                    nums, target, _ = tc
                    parsed = ast.literal_eval(actual)
                    if isinstance(parsed, tuple):
                        parsed = list(parsed)
                    if (
                        isinstance(parsed, list)
                        and len(parsed) == 2
                        and all(isinstance(x, int) for x in parsed)
                        and parsed[0] != parsed[1]
                        and 0 <= parsed[0] < len(nums)
                        and 0 <= parsed[1] < len(nums)
                        and nums[parsed[0]] + nums[parsed[1]] == target
                    ):
                        num_correct += 1
                elif test_name == "merge":
                    expected = tc[2]
                    parsed = ast.literal_eval(actual)
                    if list(parsed) == expected:
                        num_correct += 1
                elif test_name == "first_missing_positive":
                    expected = tc[1]
                    if int(actual) == expected:
                        num_correct += 1
                elif test_name == "merge_intervals":
                    expected = tc[1]
                    parsed = ast.literal_eval(actual)
                    parsed = [list(interval) for interval in parsed]
                    if parsed == expected:
                        num_correct += 1
                elif test_name == "decode_rle":
                    expected = tc[1]
                    if expected is ValueError:
                        if actual == "EXC:ValueError":
                            num_correct += 1
                    elif actual == expected:
                        num_correct += 1
                elif test_name == "first_occurrence":
                    expected = tc[2]
                    if int(actual) == expected:
                        num_correct += 1
            except:
                pass

        return num_correct, len(tests)
    except:
        return 0, len(tests)


def run_tests(
    port: int,
    judge_port: Optional[int] = None,
    categories: list[str] = None,
    use_llm_judge: bool = True,
) -> TestSuiteResult:
    """Run test suite on a running model."""
    if categories is None:
        categories = ["code", "debugging"]

    categories_dict = discover_tests()
    results = []

    for category in categories:
        if category not in categories_dict:
            continue

        for test_file in categories_dict[category]:
            test_name = test_file.stem
            prompt = test_file.read_text().strip()

            print(f"  Running {category}/{test_name}...")

            # Run test
            response = run_chat_completion(port, prompt)

            ran, correct, score, output = _evaluate_single_response(
                test_name, category, response
            )
            if category in ["code", "debugging"]:
                print(f"    ran={ran}")
                print(f"    correct={correct}")
                if output:
                    print(f"    output: {output[:100]}")

            results.append(
                TestResult(
                    name=test_name,
                    category=category,
                    prompt=prompt,
                    response=response,
                    ran=ran,
                    correct=correct,
                    score=score,
                )
            )

    # Calculate overall score
    scores = [r.score for r in results if r.score is not None]
    overall = sum(scores) / len(scores) if scores else 0.0

    return TestSuiteResult(
        model_name="",
        backend="",
        results=results,
        overall_score=overall,
    )


def generate_online_eval_prompt(results: list[TestResult]) -> str:
    """Generate copy/pasteable prompt for online LLM evaluation."""
    prompt = """# Model Evaluation

Please evaluate the following model responses. For each test, rate the response 1-10 and explain your reasoning.

## Evaluation Criteria:
- Correctness and completeness
- Code quality and functionality
- Following instructions accurately

---

"""

    for i, r in enumerate(results, 1):
        prompt += f"## Test {i}: {r.name} ({r.category})\n\n"
        prompt += f"**Prompt:**\n{r.prompt}\n\n"
        prompt += f"**Response:**\n{r.response}\n\n"
        prompt += "**Score:** __/10\n\n"
        prompt += "---\n\n"

    return prompt


def save_results_json(result: TestSuiteResult, path: Path) -> None:
    """Save results to JSON."""
    data = {
        "model_name": result.model_name,
        "backend": result.backend,
        "overall_score": result.overall_score,
        "tests": [
            {
                "name": r.name,
                "category": r.category,
                "prompt": r.prompt,
                "response": r.response,
                "ran": r.ran,
                "correct": r.correct,
                "score": r.score,
            }
            for r in result.results
        ],
    }
    path.write_text(json.dumps(data, indent=2))


def save_results_csv(results: list[dict], path: Path) -> None:
    """Save batch results to CSV with one aggregate column per task type."""
    import csv

    def aggregate_category_score(tests: list[dict], category: str) -> str:
        category_tests = [t for t in tests if t.get("category") == category]
        if not category_tests:
            return ""

        total_score = sum(float(t.get("score", 0) or 0) for t in category_tests)
        max_score = 10 * len(category_tests)
        return f"{total_score / max_score:.4f}"

    header = [
        "model",
        "backend",
        "ctk",
        "ctv",
        "ctx",
        "test_time_seconds",
        "code",
        "debug",
    ]
    rows = []
    for r in results:
        tests = r.get("tests", [])
        row = [
            r.get("model", ""),
            r.get("backend", ""),
            r.get("ctk", ""),
            r.get("ctv", ""),
            r.get("ctx", ""),
            r.get("test_time_seconds", ""),
            aggregate_category_score(tests, "code"),
            aggregate_category_score(tests, "debugging"),
        ]
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def reevaluate_batch_results_file(path: Path) -> list[dict]:
    """Recompute ran/correct/score fields in an existing batch results JSON file."""
    data = json.loads(path.read_text())
    for run in data:
        for test in run.get("tests", []):
            ran, correct, score, _ = _evaluate_single_response(
                test.get("name", ""),
                test.get("category", ""),
                test.get("response", ""),
            )
            test["ran"] = ran
            test["correct"] = correct
            test["score"] = score

        tests = run.get("tests", [])
        if tests:
            run["score"] = sum(float(t.get("score", 0) or 0) for t in tests) / len(
                tests
            )
        else:
            run["score"] = 0.0
    path.write_text(json.dumps(data, indent=2))
    return data


def load_batch_results(json_path: Path) -> list[dict]:
    """Load batch results from JSON."""
    return json.loads(json_path.read_text())


def export_batch_csv(json_path: Path, csv_path: Path) -> None:
    """Export batch JSON results to CSV."""
    results = load_batch_results(json_path)
    save_results_csv(results, csv_path)


def format_markdown(result: TestSuiteResult) -> str:
    """Format results as markdown table."""
    lines = [
        f"# Test Results: {result.model_name}",
        f"Backend: {result.backend}",
        f"Overall Score: {result.overall_score:.1f}/10",
        "",
        "| Test | Category | Score |",
        "|------|----------|-------|",
    ]

    for r in result.results:
        score_str = f"{r.score}/10" if r.score is not None else "N/A"
        lines.append(f"| {r.name} | {r.category} | {score_str} |")

    return "\n".join(lines)
