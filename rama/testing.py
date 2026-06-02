import csv
import json
import subprocess
import sys
from pathlib import Path

from .log import setup_logging
from .testing_modules.base import TestResult, TestSuiteResult
from .testing_modules.code_runner import load_task, run_task

logger = setup_logging()
PYTHON_BIN = sys.executable or "python"


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


def run_chat_completion(
    port: int,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    seed: int = 42,
    retries: int = 2,
    host: str = "127.0.0.1",
    system_prompt: str | None = None,
    conversation: list[dict] | None = None,
    top_p: float = 1.0,
    top_k: int | None = None,
    min_p: float | None = None,
    timeout_seconds: int = 120,
) -> str:
    """Call the model via llama.cpp API with retry support.

    If conversation is provided, sends the full message history
    (system + previous exchanges + new user prompt) to maintain
    proper OpenAI-style conversation continuity.
    """
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/v1/chat/completions"
    if conversation is not None:
        messages = list(conversation)
        messages.append({"role": "user", "content": prompt})
    else:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "top_p": top_p,
        "stream": False,
    }
    if top_k is not None:
        data["top_k"] = top_k
    if min_p is not None:
        data["min_p"] = min_p

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
                choice = result.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                if content == "":
                    logger.warning(
                        "Empty completion received (finish_reason=%s, choice_keys=%s)",
                        choice.get("finish_reason"),
                        sorted(choice.keys()),
                    )
                return content
        except urllib.error.HTTPError as e:
            logger.error(
                "HTTP error on attempt %d: %d %s", attempt + 1, e.code, e.reason
            )
            if attempt == retries:
                return f"ERROR: {e.code} {e.reason}"
        except Exception as e:
            logger.error("Error on attempt %d: %s", attempt + 1, e)
            if attempt == retries:
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
            logger.debug(f"HTTP warmup request failed: {e}")


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
        logger.debug(f"context reset failed: {e}")
        return {"conversation": [{"role": "system", "content": system_prompt}]}


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
        judge_port,
        judge_prompt,
        temperature=0.0,
        max_tokens=5,
        seed=42,
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

def run_tests(
    port: int,
    judge_port: int | None = None,
    categories: list[str] | None = None,
    use_llm_judge: bool = True,
    config=None,
    system_prompt: str | dict[str, str] | None = None,
    warmup_rounds: int = 0,
    context_text: str | None = None,
    context_files: list[str] | None = None,
    context_placement: str = "before",
    temperature: float = 0.0,
    seed: int = 42,
    top_p: float = 1.0,
    top_k: int | None = None,
    min_p: float | None = None,
    max_tokens: int = 1024,
    timeout_seconds: int = 300,
    conversation: list[dict] | None = None,
) -> TestSuiteResult:
    """Run test suite on a running model.

    Args:
        port: Port for the model API
        judge_port: Optional port for LLM judge
        categories: Test categories to run
        use_llm_judge: Whether to use LLM judge for scoring
        config: Rama config dict
        system_prompt: Optional system prompt
        warmup_rounds: Number of warm-up rounds before tests
        context_text: Wiki text to inject into context
        context_files: List of code file contents to inject
        context_placement: Where to place context ("before", "after", "interleaved", "before_after")
    """
    if categories is None:
        categories = ["code", "debugging"]

    host = config["host"] if config else "127.0.0.1"

    if warmup_rounds > 0:
        logger.info("Warming up model with %d rounds...", warmup_rounds)
        warmup_model(port, host=host, messages=warmup_rounds)

    categories_dict = discover_tests(config)
    results = []

    # Build context if provided
    context_parts = []
    if context_text:
        context_parts.append(context_text)
    if context_files:
        context_parts.extend(context_files)
    full_context = "\n\n".join(context_parts) if context_parts else None

    for category in categories:
        if category not in categories_dict:
            continue

        for test_file in categories_dict[category]:
            task = load_task(test_file)
            test_name = test_file.stem
            prompt = task["prompt"]

            # Build final prompt with context injection
            final_prompt = _build_prompt_with_context(
                prompt, full_context, context_placement
            )
            category_system_prompt = (
                system_prompt.get(category)
                if isinstance(system_prompt, dict)
                else system_prompt
            )

            # Run test
            response = run_chat_completion(
                port,
                final_prompt,
                host=host,
                system_prompt=category_system_prompt,
                temperature=temperature,
                seed=seed,
                conversation=conversation,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                timeout_seconds=timeout_seconds,
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
        "seed",
        "run_type",
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
            r.get("seed", ""),
            r.get("run_type", ""),
            r.get("test_time_seconds", ""),
            aggregate_category_score(tests, "code"),
            aggregate_category_score(tests, "debugging"),
        ]
        rows.append(row)

    with path.open("w", newline="") as f:
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
                tests,
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
