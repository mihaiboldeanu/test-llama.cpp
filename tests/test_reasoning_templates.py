"""Tests for family-specific reasoning template controls."""

import json
import unittest
from unittest.mock import patch

from rama import Config
from rama.testing import ChatCompletionConfig, run_chat_completion


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"choices":[{"message":{"content":"ok"}}]}'


class ReasoningTemplateTests(unittest.TestCase):
    def test_north_uses_its_reasoning_template_key(self) -> None:
        config = ChatCompletionConfig(
            port=1,
            prompt="hi",
            reasoning=False,
            reasoning_template_key="reasoning",
            presence_penalty=1.5,
            repeat_penalty=1.0,
        )

        with patch("urllib.request.urlopen", return_value=_Response()) as mocked:
            self.assertEqual(run_chat_completion(config), "ok")

        request = mocked.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["chat_template_kwargs"], {"reasoning": False})
        self.assertEqual(payload["presence_penalty"], 1.5)
        self.assertEqual(payload["repeat_penalty"], 1.0)

    def test_gpt_oss_uses_reasoning_effort_without_boolean_reasoning(self) -> None:
        config = ChatCompletionConfig(
            port=1,
            prompt="hi",
            reasoning=False,
            reasoning_template_key="reasoning_effort",
            reasoning_template_value="low",
        )

        with patch("urllib.request.urlopen", return_value=_Response()) as mocked:
            self.assertEqual(run_chat_completion(config), "ok")

        request = mocked.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["chat_template_kwargs"], {"reasoning_effort": "low"})
        self.assertNotIn("reasoning", payload)

    def test_gpt_oss_skips_native_chat_parsing(self) -> None:
        config = Config("rama.yaml")
        self.assertTrue(config.families["gpt-oss"]["skip_chat_parsing"])


if __name__ == "__main__":
    unittest.main()
