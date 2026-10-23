"""Regression tests for model-family detection of supported downloads."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from rama.core import (
    detect_family,
    detect_quant,
    discover_models,
)
from rama.launch import _resolve_draft_model_path


class ModelFamilyTests(unittest.TestCase):
    def test_qwen_coder_is_not_collapsed_into_generic_qwen(self) -> None:
        self.assertEqual(
            detect_family(Path("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf")),
            "qwen-coder",
        )

    def test_gpt_oss_is_detected(self) -> None:
        self.assertEqual(
            detect_family(Path("gpt-oss-20b-mxfp4.gguf")),
            "gpt-oss",
        )

    def test_lfm_and_ling_are_detected(self) -> None:
        self.assertEqual(detect_family(Path("LFM2-24B-A2B-Q8_0.gguf")), "lfm")
        self.assertEqual(detect_family(Path("Ling-mini-2.0-Q6_K.gguf")), "ling")

    def test_flint_is_detected(self) -> None:
        self.assertEqual(detect_family(Path("flint-Q8_0.gguf")), "flint")
        self.assertEqual(
            detect_family(Path("Flint-Preview-UD-Q4_K_XL.gguf")), "flint"
        )

    def test_bonsai_27_and_nex_n2_are_detected(self) -> None:
        self.assertEqual(
            detect_family(Path("Ternary-Bonsai-27B-Q2_0.gguf")), "bonsai-27"
        )
        self.assertEqual(detect_family(Path("Nex-N2-mini-Q4_K_M.gguf")), "nex-n2")
        self.assertEqual(detect_quant("Bonsai-27B-Q1_0.gguf"), "Q1")
        self.assertEqual(detect_quant("Ternary-Bonsai-27B-Q2_0.gguf"), "Q2")

    def test_grug_uses_the_qwen_compatible_family(self) -> None:
        self.assertEqual(detect_family(Path("grug-27b-Q4_K_M.gguf")), "grug")

    def test_spark_x25_is_detected(self) -> None:
        self.assertEqual(
            detect_family(Path("Spark-X2.5-4B-Q8_0.gguf")), "spark"
        )
        self.assertEqual(
            detect_family(Path("Spark-X2.5-1.7B-Q4_K_M.gguf")), "spark"
        )

    def test_draft_model_name_resolves_to_its_gguf(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            draft = model_dir / "dflash-kquant-muse.gguf"
            draft.touch()
            config = SimpleNamespace(model_dir=model_dir)

            self.assertEqual(
                _resolve_draft_model_path(config, "dflash"), str(draft)
            )

    def test_complete_split_gguf_is_one_runnable_model(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            for index in range(1, 4):
                (model_dir / f"Laguna-S-2.1-IQ4_XS-{index:05d}-of-00003.gguf").write_bytes(
                    bytes([index])
                )

            models = discover_models(SimpleNamespace(model_dir=model_dir))

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "Laguna-S-2.1-IQ4_XS")
        self.assertEqual(models[0].family, "laguna")
        self.assertEqual(models[0].size_gb, 0.0)

    def test_qwen38_flash_next_split_gguf_is_detected_as_one_model(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            for index in range(1, 4):
                (
                    model_dir
                    / f"Qwen3.8-Flash-Next-UD-IQ3_XXS-{index:05d}-of-00003.gguf"
                ).write_bytes(bytes([index]))

            models = discover_models(SimpleNamespace(model_dir=model_dir))

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "Qwen3.8-Flash-Next-UD-IQ3_XXS")
        self.assertEqual(models[0].family, "qwen3.8-flash-next")

    def test_incomplete_split_gguf_is_not_listed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "Laguna-S-2.1-IQ4_XS-00001-of-00003.gguf").touch()
            (model_dir / "Laguna-S-2.1-IQ4_XS-00003-of-00003.gguf").touch()

            models = discover_models(SimpleNamespace(model_dir=model_dir))

        self.assertEqual(models, [])



if __name__ == "__main__":
    unittest.main()
