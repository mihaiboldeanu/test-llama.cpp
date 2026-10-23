"""Tests for batch config loading (defaults:/models: shape)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rama.domain.batch_planner import load_batch_items


class LoadBatchItemsTests(unittest.TestCase):
    def _write(self, directory: str, content: str) -> Path:
        path = Path(directory) / "batch.yaml"
        path.write_text(content)
        return path

    def test_defaults_merge_into_string_and_dict_items(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(
                directory,
                "defaults:\n"
                "  ctx: 16384\n"
                "  ctk: f16\n"
                "models:\n"
                "  - model-a\n"
                "  - model: model-b\n",
            )
            items = load_batch_items(path)

        self.assertEqual(
            items,
            [
                {"model": "model-a", "ctx": 16384, "ctk": "f16"},
                {"model": "model-b", "ctx": 16384, "ctk": "f16"},
            ],
        )

    def test_per_item_keys_win_over_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(
                directory,
                "defaults:\n"
                "  ctx: 16384\n"
                "  reasoning: false\n"
                "models:\n"
                "  - model: model-a\n"
                "    ctx: 32768\n",
            )
            items = load_batch_items(path)

        self.assertEqual(
            items, [{"model": "model-a", "ctx": 32768, "reasoning": False}]
        )

    def test_legacy_bare_list_passes_through(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(
                directory,
                "- model: model-a\n"
                "  ctx: 16384\n"
                '- "model-b"\n',
            )
            items = load_batch_items(path)

        self.assertEqual(items, [{"model": "model-a", "ctx": 16384}, "model-b"])

    def test_empty_file_gives_no_items(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(directory, "")
            self.assertEqual(load_batch_items(path), [])

    def test_mapping_without_models_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(directory, "defaults:\n  ctx: 16384\n")
            with self.assertRaises(ValueError):
                load_batch_items(path)

    def test_all_shipped_configs_keep_their_model_counts(self) -> None:
        config_dir = Path(__file__).parent.parent / "config"
        expected = {
            "batch_code_test.yaml": 32,
            "batch_code_test_reasoning.yaml": 32,
            "batch_small.yaml": 11,
            "batch_ppl.yaml": 12,
        }
        for name, count in expected.items():
            with self.subTest(config=name):
                items = load_batch_items(config_dir / name)
                self.assertEqual(len(items), count)
                for item in items:
                    self.assertIn("model", item)


if __name__ == "__main__":
    unittest.main()
