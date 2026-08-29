"""Reference-solver regressions for repaired benchmark fixtures."""

import json
import textwrap
import unittest
from pathlib import Path

from rama.testing_modules.code_runner import run_task

FIXTURES = Path(__file__).parent


def load_fixture(path: str) -> dict:
    return json.loads((FIXTURES / path).read_text())


class FixtureRegressionTests(unittest.TestCase):
    def assert_reference_passes(self, path: str, code: str) -> None:
        result = run_task(textwrap.dedent(code), load_fixture(path))
        self.assertTrue(result["passed"], result["details"])

    def test_top_k_fixture_agrees_with_its_sampling_contract(self) -> None:
        self.assert_reference_passes(
            "code/top_k_frequent.json",
            """
            from collections import Counter

            def top_k_frequent(nums, k):
                counts = Counter(nums)
                return sorted(counts, key=lambda value: (-counts[value], value))[:k]
            """,
        )

    def test_candy_crush_debug_fixture_uses_gravity_for_a_stable_board(self) -> None:
        self.assert_reference_passes(
            "debugging/candy_crush_debug.json",
            """
            def candy_crush(board):
                rows, cols = len(board), len(board[0])
                while True:
                    crushed = set()
                    for row in range(rows):
                        for col in range(cols - 2):
                            value = board[row][col]
                            if value and value == board[row][col + 1] == board[row][col + 2]:
                                crushed.update((row, col + offset) for offset in range(3))
                    for col in range(cols):
                        for row in range(rows - 2):
                            value = board[row][col]
                            if value and value == board[row + 1][col] == board[row + 2][col]:
                                crushed.update((row + offset, col) for offset in range(3))
                    if not crushed:
                        return board
                    for row, col in crushed:
                        board[row][col] = 0
                    for col in range(cols):
                        values = [board[row][col] for row in range(rows) if board[row][col]]
                        for row in range(rows):
                            board[row][col] = 0
                        for row, value in enumerate(values, rows - len(values)):
                            board[row][col] = value
            """,
        )

    def test_rotate_fixture_requires_the_documented_return_value(self) -> None:
        self.assert_reference_passes(
            "debugging/rotate_matrix.json",
            """
            def rotate(matrix):
                matrix[:] = [list(row) for row in zip(*matrix[::-1])]
                return matrix
            """,
        )

    def test_substring_fixture_uses_valid_concatenation_indices(self) -> None:
        self.assert_reference_passes(
            "debugging/substring_concatenation_debug.json",
            """
            from collections import Counter

            def find_substring(text, words):
                if not text or not words:
                    return []
                width = len(words[0])
                total = width * len(words)
                wanted = Counter(words)
                return [
                    start
                    for start in range(len(text) - total + 1)
                    if Counter(text[index:index + width] for index in range(start, start + total, width)) == wanted
                ]
            """,
        )

    def test_word_search_fixture_uses_orthogonal_paths_and_unique_words(self) -> None:
        self.assert_reference_passes(
            "debugging/word_search_ii_debug.json",
            """
            def find_words(board, words):
                if not board or not words:
                    return []
                wanted = set(words)
                prefixes = {word[:length] for word in wanted for length in range(1, len(word) + 1)}
                rows, cols = len(board), len(board[0])
                found = set()

                def search(row, col, path, visited):
                    if not (0 <= row < rows and 0 <= col < cols) or (row, col) in visited:
                        return
                    path += board[row][col]
                    if path not in prefixes:
                        return
                    if path in wanted:
                        found.add(path)
                    visited.add((row, col))
                    for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        search(row + delta_row, col + delta_col, path, visited)
                    visited.remove((row, col))

                for row in range(rows):
                    for col in range(cols):
                        search(row, col, "", set())
                return list(found)
            """,
        )


if __name__ == "__main__":
    unittest.main()
