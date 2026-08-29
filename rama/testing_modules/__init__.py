"""Unified testing harness modules for model evaluation."""

from .base import TestModule, TestResult, TestSuiteResult
from .context_difficulty import ContextDifficultyConfig, ContextDifficultyModule
from .enhanced_nihs import EnhancedNIHSConfig, EnhancedNIHSModule
from .system_prompt import (
    SYSTEM_PROMPT_TEMPLATES,
    get_code_system_prompt,
    get_debug_system_prompt,
    get_system_prompt,
)

__all__ = [
    "SYSTEM_PROMPT_TEMPLATES",
    "ContextDifficultyConfig",
    "ContextDifficultyModule",
    "EnhancedNIHSConfig",
    "EnhancedNIHSModule",
    "TestModule",
    "TestResult",
    "TestSuiteResult",
    "get_code_system_prompt",
    "get_debug_system_prompt",
    "get_system_prompt",
]
