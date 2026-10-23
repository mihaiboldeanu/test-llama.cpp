import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .log import setup_logging
from .testing_modules.base import TestResult, TestSuiteResult
from .testing_modules.code_runner import load_task, run_task

logger = setup_logging()


_SAMPLER_ARG_FIELDS = {
    "--temp": ("temperature", float),
    "--top-p": ("top_p", float),
    "--top-k": ("top_k", int),
    "--min-p": ("min_p", float),
    "--presence-penalty": ("presence_penalty", float),
    "--repeat-penalty": ("repeat_penalty", float),
}


def sampler_fields_from_args(args: dict | None) -> dict:
    """Translate merged llama-server sampler flags to chat-request fields."""
    fields: dict = {}
    for flag, (field, convert) in _SAMPLER_ARG_FIELDS.items():
        if args is not None and args.get(flag) is not None:
            try:
                fields[field] = convert(args[flag])
            except (TypeError, ValueError):
                continue
    return fields


def get_tests_dir(config=None) -> Path:
    """Get the tests directory from config or default."""
    if config and config.get("tests_dir"):
        p = Path(config["tests_dir"])
        if p.exists():
            return p
    return Path(__file__).parent.parent / "tests"


def discover_tests(config=None) -> dict[str, list[Path]]:
    """Discover all test files by category."""
    tests_dir = get_tests_dir(config)
    categories = {}

    for category in ["code", "debugging"]:
        cat_dir = tests_dir / category
        if cat_dir.exists():
            categories[category] = sorted(cat_dir.glob("*.json"))

    return categories


@dataclass
class ChatCompletionConfig:
    """Configuration for a chat completion request.

    Sampler fields can be derived from merged launch args via the optional
    ``sampler_args`` mapping, or set individually for backward compatibility.
    When provided, ``sampler_args`` overrides individual sampler fields.
    """

    port: int
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int = 42
    retries: int = 2
    host: str = "127.0.0.1"
    system_prompt: str | None = None
    conversation: list[dict] | None = None
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None
    timeout_seconds: int = 120
    reasoning: bool | None = None
    reasoning_template_key: str = "enable_thinking"
    reasoning_template_value: bool | str | None = None

    # New profile-derived fields (override individual fields when set)
    sampler_args: dict | None = None

    def __post_init__(self):
        for field, value in sampler_fields_from_args(self.sampler_args).items():
            setattr(self, field, value)


def run_chat_completion(cfg: ChatCompletionConfig) -> str:
    """Call the model via llama.cpp API with retry support."""
    import urllib.error
    import urllib.request

    url = f"http://{cfg.host}:{cfg.port}/v1/chat/completions"
    if cfg.conversation is not None:
        messages = list(cfg.conversation)
        messages.append({"role": "user", "content": cfg.prompt})
    else:
        messages = [{"role": "user", "content": cfg.prompt}]
        if cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.system_prompt})
    data = {
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "seed": cfg.seed,
        "top_p": cfg.top_p,
        "stream": False,
    }
    if cfg.top_k is not None:
        data["top_k"] = cfg.top_k
    if cfg.min_p is not None:
        data["min_p"] = cfg.min_p
    if cfg.presence_penalty is not None:
        data["presence_penalty"] = cfg.presence_penalty
    if cfg.repeat_penalty is not None:
        data["repeat_penalty"] = cfg.repeat_penalty
    if cfg.reasoning is not None:
        # llama.cpp controls this through template parameters. Families select
        # the key their embedded template actually understands.
        template_value = (
            cfg.reasoning_template_value
            if cfg.reasoning_template_value is not None
            else cfg.reasoning
        )
        data["chat_template_kwargs"] = {
            cfg.reasoning_template_key: (template_value),
        }
        if isinstance(template_value, bool):
            data["reasoning"] = cfg.reasoning

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    for attempt in range(cfg.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                result = json.loads(raw)
                choice = result.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                if content == "" or "message" not in choice:
                    logger.warning(
                        "Empty/missing completion (finish_reason=%s, has_message=%s, raw_choice=%s)",
                        choice.get("finish_reason"),
                        "message" in choice,
                        raw[:500],
                    )
                return content
        except urllib.error.HTTPError as e:
            logger.error(
                "HTTP error on attempt %d: %d %s", attempt + 1, e.code, e.reason
            )
            logger.debug("Request data: %s", data)
            if attempt == cfg.retries:
                return f"ERROR: {e.code} {e.reason}"
        except Exception as e:
            logger.error("Error on attempt %d: %s", attempt + 1, e)
            logger.debug("Request data: %s", data)
            if attempt == cfg.retries:
                return f"ERROR: {e!s}"

    return "ERROR: Max retries exceeded"


def _build_prompt_with_context(
    prompt: str,
    context: str | None,
    placement: str = "before",
) -> str:
    """Build prompt with injected context based on placement strategy.

    Args:
        prompt: The original test prompt
        context: Context text to inject (wiki + code files)
        placement: Where to place context ("before", "after", "interleaved", "before_after")
    """
    if not context:
        return prompt

    if placement == "before":
        return f"{context}\n\n---\n\n{prompt}"

    if placement == "after":
        return f"{prompt}\n\n---\n\nContext:\n{context}"

    if placement == "before_after":
        return f"{context}\n\n---\n\n{prompt}\n\n---\n\nNote: The context above may contain useful or irrelevant information."

    if placement == "interleaved":
        # Split context into chunks and interleave with prompt
        chars = list(context)
        chunk_size = max(1, len(chars) // 4)
        chunks = [
            "".join(chars[i : i + chunk_size]) for i in range(0, len(chars), chunk_size)
        ]

        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(chunk)
            if i == len(chunks) // 2:
                parts.append(f"\n\n---\n\n{prompt}\n\n---\n\n")
        parts.append("".join(chunks[len(chunks) // 2 :]))

        return "\n\n".join(parts)

    # Default: before
    return f"{context}\n\n---\n\n{prompt}"


def warmup_model(port: int, host: str = "127.0.0.1", messages: int = 3) -> None:
    """Warm up the model by sending a few short messages.

    Follows llama.cpp bench best practices: discard first N iterations,
    use a 3-message protocol to establish KV cache state.
    """
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/v1/chat/completions"
    short_prompts = [
        "Hello, can you hear me?",
        "Yes, I can hear you clearly. How can I help?",
        "Thank you. That is all for now.",
    ]

    for i in range(messages):
        prompt = short_prompts[i % len(short_prompts)]
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
            "stream": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                response.read()
        except Exception as e:
            logger.debug("HTTP warmup request failed: %s", e)


def reset_context(port: int, system_prompt: str, host: str = "127.0.0.1") -> dict:
    """Reset the model context by sending system prompt + short exchange.

    Sends system_prompt -> user "Hello" -> captures assistant response.
    Returns the full conversation list that should be sent with subsequent
    requests to maintain proper OpenAI-style conversation history.

    Returns dict with 'conversation' key containing the message list.
    """
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/v1/chat/completions"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hello, can you hear me?"},
    ]
    data = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            assistant_content = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            # Return full conversation including assistant response
            messages.append({"role": "assistant", "content": assistant_content})
            return {"conversation": messages}
    except Exception as e:
        # If reset fails, return minimal conversation
        logger.debug("context reset failed: %s", e)
        return {"conversation": [{"role": "system", "content": system_prompt}]}


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
        r"```([^\n`]*)\n?(.*?)(?:```|$)",
        response,
        re.DOTALL | re.IGNORECASE,
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
        # If no preferred language, use the first block with Python-like content
        python_like = [
            (lang, body)
            for lang, body in fenced_blocks
            if "import " in body or "def " in body or "class " in body
        ]
        if python_like:
            return _normalize_code(python_like[0][1])
        return _normalize_code(fenced_blocks[0][1])

    lines = response.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ", "class ", "def ", "@")):
            return _normalize_code("\n".join(lines[i:]))

    return _normalize_code(response)


def _evaluate_single_response(
    test_name: str,
    category: str,
    response: str,
    task: dict | None = None,
) -> tuple[bool, bool, int, str]:
    """Evaluate one response using the data-driven code runner."""
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

    code = _extract_code(response)

    if task is not None:
        result = run_task(code, task)
        correct = result["passed"]
        score = 10 if correct else 0
    else:
        correct = False
        score = 0

    return ran, correct, score, output


def _format_test_status(ran: bool, correct: bool, score: int, response: str) -> str:
    """Format a concise per-test status string for progress logs."""
    if response.startswith("ERROR:"):
        return f"ERROR {score}/10"
    if not response.strip():
        return "EMPTY 0/10"
    if not ran:
        return f"NO-RUN {score}/10"
    if correct:
        return f"PASS {score}/10"
    return f"FAIL {score}/10"


def summarize_evaluation(results: list[TestResult]) -> dict:
    """Turn raw test cases into a comparable model-quality verdict.

    A high score is only useful when the model consistently returns an answer
    that can be executed.  Keep those signals separate so template or server
    failures are never mistaken for a weak model.
    """
    total = len(results)
    if total == 0:
        return {
            "verdict": "no data",
            "score": None,
            "tests": 0,
            "executed": 0,
            "completion_rate": 0.0,
            "execution_rate": 0.0,
            "category_scores": {},
            "recurring_failures": [],
        }

    scored = [result.score for result in results if result.score is not None]
    score = sum(scored) / len(scored) if scored else None
    completed = [
        result
        for result in results
        if result.response.strip() and not result.response.startswith("ERROR:")
    ]
    executed = [result for result in results if result.ran]
    completion_rate = len(completed) / total
    execution_rate = len(executed) / total

    category_scores = {}
    for category in sorted({result.category for result in results if result.category}):
        category_results = [
            result.score
            for result in results
            if result.category == category and result.score is not None
        ]
        if category_results:
            category_scores[category] = round(
                sum(category_results) / len(category_results), 2
            )

    failure_counts: dict[tuple[str, str], int] = {}
    for result in results:
        failed = (
            not result.response.strip()
            or result.response.startswith("ERROR:")
            or not result.ran
            or not result.correct
        )
        if failed:
            key = (result.category or "uncategorized", result.name)
            failure_counts[key] = failure_counts.get(key, 0) + 1
    recurring_failures = [
        {"category": category, "name": name, "count": count}
        for (category, name), count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0])
        )[:5]
    ]

    if completion_rate < 0.90 or execution_rate < 0.75:
        verdict = "broken"
    elif (
        score is not None
        and score >= 8.0
        and completion_rate >= 0.98
        and execution_rate >= 0.98
    ):
        verdict = "excellent"
    elif score is not None and score >= 7.0:
        verdict = "good"
    elif score is not None and score >= 5.5:
        verdict = "usable with review"
    else:
        verdict = "not recommended"

    return {
        "verdict": verdict,
        "score": round(score, 2) if score is not None else None,
        "tests": total,
        "executed": len(executed),
        "completion_rate": round(completion_rate, 4),
        "execution_rate": round(execution_rate, 4),
        "category_scores": category_scores,
        "recurring_failures": recurring_failures,
    }


@dataclass
class TestRunConfig:
    """Configuration for running a test suite."""

    port: int
    judge_port: int | None = None
    categories: list[str] | None = None
    use_llm_judge: bool = True
    config: dict | None = None
    system_prompt: str | dict[str, str] | None = None
    warmup_rounds: int = 0
    context_text: str | None = None
    context_files: list[str] | None = None
    context_placement: str = "before"
    temperature: float = 0.0
    seed: int = 42
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None
    max_tokens: int = 2048
    timeout_seconds: int = 300
    conversation: list[dict] | None = None
    reasoning: bool | None = None
    reasoning_template_key: str = "enable_thinking"
    reasoning_template_value: bool | str | None = None
    max_tests_per_category: int | None = None

    # Profile-derived fields (from ResolvedProfile.args)
    sampler_args: dict | None = None

    def __post_init__(self):
        for field, value in sampler_fields_from_args(self.sampler_args).items():
            setattr(self, field, value)


def run_tests(cfg: TestRunConfig) -> TestSuiteResult:
    """Run test suite on a running model."""
    categories = cfg.categories or ["code", "debugging"]
    host = (cfg.config or {})["host"] if cfg.config else "127.0.0.1"

    if cfg.warmup_rounds > 0:
        logger.info("Warming up model with %d rounds...", cfg.warmup_rounds)
        warmup_model(cfg.port, host=host, messages=cfg.warmup_rounds)

    categories_dict = discover_tests(cfg.config)
    results = []

    context_parts = []
    if cfg.context_text:
        context_parts.append(cfg.context_text)
    if cfg.context_files:
        context_parts.extend(cfg.context_files)
    full_context = "\n\n".join(context_parts) if context_parts else None

    for category in categories:
        if category not in categories_dict:
            continue

        test_files = categories_dict[category]
        if cfg.max_tests_per_category is not None:
            test_files = test_files[: cfg.max_tests_per_category]

        for test_file in test_files:
            task = load_task(test_file)
            test_name = test_file.stem
            prompt = task["prompt"]

            final_prompt = _build_prompt_with_context(
                prompt, full_context, cfg.context_placement
            )
            category_system_prompt = (
                cfg.system_prompt.get(category)
                if isinstance(cfg.system_prompt, dict)
                else cfg.system_prompt
            )

            response = run_chat_completion(
                ChatCompletionConfig(
                    port=cfg.port,
                    prompt=final_prompt,
                    host=host,
                    system_prompt=category_system_prompt,
                    temperature=cfg.temperature,
                    seed=cfg.seed,
                    conversation=cfg.conversation,
                    max_tokens=cfg.max_tokens,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    min_p=cfg.min_p,
                    presence_penalty=cfg.presence_penalty,
                    repeat_penalty=cfg.repeat_penalty,
                    timeout_seconds=cfg.timeout_seconds,
                    reasoning=cfg.reasoning,
                    reasoning_template_key=cfg.reasoning_template_key,
                    reasoning_template_value=cfg.reasoning_template_value,
                    sampler_args=cfg.sampler_args,
                )
            )

            ran, correct, score, output = _evaluate_single_response(
                test_name,
                category,
                response,
                task,
            )
            if category in ["code", "debugging"]:
                logger.debug("ran=%s", ran)
                logger.debug("correct=%s", correct)
                if output:
                    logger.debug("output: %s", output[:100])
            logger.info(
                "%s/%s %s",
                category,
                test_name,
                _format_test_status(ran, correct, score, response),
            )

            results.append(
                TestResult(
                    name=test_name,
                    category=category,
                    prompt=prompt,
                    response=response,
                    ran=ran,
                    correct=correct,
                    score=score,
                ),
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





def save_results_csv(results: list[dict], path: Path) -> None:
    """Save one sortable summary row per model for a batch result."""

    def aggregate_category_score(tests: list[dict], category: str) -> str:
        category_tests = [test for test in tests if test.get("category") == category]
        if not category_tests:
            return ""

        total_score = sum(float(t.get("score", 0) or 0) for t in category_tests)
        max_score = 10 * len(category_tests)
        return f"{total_score / max_score:.4f}"

    def aggregate_run_scores(runs: list[dict], run_type: str) -> tuple[str, str]:
        scores = [
            float(run["score"])
            for run in runs
            if run.get("run_type") == run_type and run.get("score") is not None
        ]
        if not scores:
            return "", ""
        mean = sum(scores) / len(scores)
        stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        return f"{mean:.4f}", f"{stddev:.4f}"

    def aggregate_score(result: dict, tests: list[dict]) -> str:
        if result.get("score") is not None:
            return f"{float(result['score']):.4f}"
        scores = [
            float(test["score"]) for test in tests if test.get("score") is not None
        ]
        return f"{sum(scores) / len(scores):.4f}" if scores else ""

    header = [
        "model",
        "backend",
        "ctk",
        "ctv",
        "ctx",
        "reasoning",
        "score",
        "score_stddev",
        "verdict",
        "completion_rate",
        "execution_rate",
        "runs",
        "sampled_runs",
        "family_score",
        "family_score_stddev",
        "test_time_seconds",
        "code",
        "debug",
    ]
    rows = []
    for r in results:
        tests = r.get("tests", [])
        runs = r.get("runs", [])
        evaluation = r.get("evaluation", {})
        family_score, family_stddev = aggregate_run_scores(runs, "family")
        row = [
            r.get("model", ""),
            r.get("backend", ""),
            r.get("ctk", ""),
            r.get("ctv", ""),
            r.get("ctx", ""),
            r.get("reasoning", ""),
            aggregate_score(r, tests),
            r.get("score_stddev", ""),
            evaluation.get("verdict", ""),
            evaluation.get("completion_rate", ""),
            evaluation.get("execution_rate", ""),
            len(runs),
            r.get("sampled_runs", ""),
            family_score,
            family_stddev,
            r.get("test_time_seconds", ""),
            aggregate_category_score(tests, "code"),
            aggregate_category_score(tests, "debugging"),
        ]
        rows.append(row)

    temp_file = path.with_suffix(f"{path.suffix}.tmp")
    with temp_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    temp_file.replace(path)


def format_markdown(result: TestSuiteResult) -> str:
    """Format results as markdown table."""
    evaluation = summarize_evaluation(result.results)
    score = (
        f"{result.overall_score:.1f}/10" if result.overall_score is not None else "N/A"
    )
    category_scores = ", ".join(
        f"{category}={score:.1f}"
        for category, score in evaluation["category_scores"].items()
    )
    lines = [
        f"# Test Results: {result.model_name}",
        f"Backend: {result.backend}",
        f"Overall Score: {score}",
        f"Verdict: {evaluation['verdict']}",
        (
            f"Reliability: completion {evaluation['completion_rate']:.0%}, "
            f"execution {evaluation['execution_rate']:.0%}"
        ),
        f"Category scores: {category_scores or 'N/A'}",
        "",
        "| Test | Category | Score |",
        "|------|----------|-------|",
    ]

    for r in result.results:
        score_str = f"{r.score}/10" if r.score is not None else "N/A"
        lines.append(f"| {r.name} | {r.category} | {score_str} |")

    return "\n".join(lines)
