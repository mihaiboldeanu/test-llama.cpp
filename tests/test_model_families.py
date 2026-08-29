"""Regression tests for model-family detection of supported downloads."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rama import Config
from rama.core import (
    ModelInfo,
    calc_kv_cache,
    detect_family,
    detect_quant,
    discover_models,
)
from rama.families import get_registry
from rama.launch import ModelStartConfig, _resolve_draft_model_path, start_model


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

    def test_bonsai_27_and_nex_n2_are_detected(self) -> None:
        self.assertEqual(
            detect_family(Path("Ternary-Bonsai-27B-Q2_0.gguf")), "bonsai-27"
        )
        self.assertEqual(detect_family(Path("Nex-N2-mini-Q4_K_M.gguf")), "nex-n2")
        self.assertEqual(detect_quant("Bonsai-27B-Q1_0.gguf"), "Q1")
        self.assertEqual(detect_quant("Ternary-Bonsai-27B-Q2_0.gguf"), "Q2")

    def test_grug_uses_the_qwen_compatible_family(self) -> None:
        self.assertEqual(detect_family(Path("grug-27b-Q4_K_M.gguf")), "grug")

    def test_muse_glimmer_uses_f16_cache_and_sizes_context_for_it(self) -> None:
        config = Config("rama.yaml")
        metadata = {
            "general.architecture": "muse-glimmer",
            "muse-glimmer.block_count": 52,
            "muse-glimmer.attention.head_count_kv": 2,
            "muse-glimmer.attention.head_dim": 208,
        }

        ctx, ctk, ctv = calc_kv_cache(
            14.8, "muse-glimmer", config, metadata=metadata
        )

        self.assertEqual((ctk, ctv), ("f16", "f16"))
        self.assertLess(ctx, 131072)
        self.assertGreaterEqual(ctx, 2048)

    def test_draft_model_name_resolves_to_its_gguf(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            draft = model_dir / "dflash-kquant-muse.gguf"
            draft.touch()
            config = SimpleNamespace(model_dir=model_dir)

            self.assertEqual(
                _resolve_draft_model_path(config, "dflash"), str(draft)
            )

    def test_muse_glimmer_preserves_high_reasoning_traces(self) -> None:
        config = Config("rama.yaml")
        family = get_registry().get("muse-glimmer")

        self.assertIsNotNone(family)
        self.assertTrue(family.capabilities.reasoning_use_preserve_flag)
        self.assertEqual(config.families["muse-glimmer"]["reasoning_enabled_value"], "high")

    def test_gemma_uses_its_embedded_chat_template(self) -> None:
        config = Config("rama.yaml")

        self.assertNotIn("chat_template_file", config.families["gemma"])

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

    def test_incomplete_split_gguf_is_not_listed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "Laguna-S-2.1-IQ4_XS-00001-of-00003.gguf").touch()
            (model_dir / "Laguna-S-2.1-IQ4_XS-00003-of-00003.gguf").touch()

            models = discover_models(SimpleNamespace(model_dir=model_dir))

        self.assertEqual(models, [])

    def test_laguna_s_uses_mmap_and_conservative_prefill_buffers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = Config("rama.yaml")
            model_path = Path(temp_dir) / "Laguna-S-2.1-UD-IQ4_XS.gguf"
            model_path.touch()
            process = Mock(pid=12345)
            process.poll.return_value = None
            model = ModelInfo(
                name="Laguna-S-2.1-UD-IQ4_XS",
                path=model_path,
                size_gb=53.6,
                family="laguna",
                quant="IQ4",
            )

            with (
                patch("rama.launch.get_server_bin", return_value=Path("/bin/true")),
                patch("rama.launch.is_port_used", return_value=False),
                patch("rama.launch.get_pid_dir", return_value=Path(temp_dir)),
                patch("rama.domain.runner.subprocess.Popen", return_value=process) as popen,
                patch("rama.domain.runner.urllib.request.urlopen"),
                patch("rama.domain.runner.time.sleep"),
            ):
                start_model(
                    ModelStartConfig(
                        model=model,
                        port=11435,
                        ctx_size=16384,
                        ctk="f16",
                        ctv="f16",
                        config=config,
                    )
                )

            args = popen.call_args.args[0]
            self.assertNotIn("--no-mmap", args)
            expected_cpu_moe = str(
                config.get("models", {})["Laguna-S-2.1-UD-IQ4_XS"]["n_cpu_moe"]
            )
            self.assertEqual(args[args.index("--n-cpu-moe") + 1], expected_cpu_moe)
            self.assertEqual(args[args.index("-b") + 1], "256")
            self.assertEqual(args[args.index("-ub") + 1], "128")



if __name__ == "__main__":
    unittest.main()
