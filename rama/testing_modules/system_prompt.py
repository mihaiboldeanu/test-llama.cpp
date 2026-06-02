"""System prompt templates for different test categories.

Templates are auto-generated based on model name and test category.
Usage:
    prompt = get_system_prompt(model_name="qwen3.6-27b", category="code")
"""

from __future__ import annotations

SYSTEM_PROMPT_TEMPLATES = {
    "code": """You are {model_name}, a senior software engineer. Write clean, well-structured
code that follows best practices. Provide only the code solution without explanations.
Focus on correctness, readability, and efficiency.""",
    "debugging": """You are {model_name}, a debugging expert. Analyze the provided code,
identify bugs, and provide the corrected version. Explain your reasoning concisely.
Focus on edge cases and error handling.""",
    "creative": """You are {model_name}, a creative writing assistant. Generate engaging,
well-structured content. Be imaginative while maintaining accuracy.""",
    "all": """You are {model_name}. You are helpful, concise, and accurate. Provide
well-structured responses that directly address the task at hand.""",
}


def get_system_prompt(
    model_name: str,
    category: str = "all",
    custom_prompt: str | None = None,
) -> str | None:
    """Get system prompt for given model and category.

    Args:
        model_name: Name of the model being tested
        category: Test category (code, debugging, creative, all)
        custom_prompt: Optional custom prompt override

    Returns:
        Formatted system prompt string, or None if no template found
    """
    if custom_prompt:
        return custom_prompt

    template = SYSTEM_PROMPT_TEMPLATES.get(category)
    if template:
        return template.format(model_name=model_name)

    return None


def get_code_system_prompt(model_name: str, custom_prompt: str | None = None) -> str | None:
    """Get system prompt for code tests."""
    return get_system_prompt(model_name, "code", custom_prompt)


def get_debug_system_prompt(model_name: str, custom_prompt: str | None = None) -> str | None:
    """Get system prompt for debugging tests."""
    return get_system_prompt(model_name, "debugging", custom_prompt)
