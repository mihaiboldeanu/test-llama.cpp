# test-llama

`test-llama` is a small CLI for running and comparing local GGUF models through `llama.cpp` and `turboquant`.

Backend repos:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
- [`turboquant_plus`](https://github.com/TheTom/turboquant_plus)

The goal is practical comparison on your own hardware. We care about which backend feels better to use, how well it handles our prompts, and how it behaves at long context lengths. This is intentionally not a perplexity/KL-divergence project.

## Install

```bash
cd test-llama.cpp
uv sync
```

If you just want to run it without a local install:

```bash
uv run test-llama <command>
```

## Configure

`test-llama` looks for config in `test_llama.yaml` in the repo root first, then in `~/.config/test_llama.yaml`. You can also pass a specific file with `--config` on any command.

Create a starter config with:

```bash
test-llama init
```

The important paths are:

- `model_dir`: where your GGUF models live
- `llama_cpp_dir`: path to your local `llama.cpp` checkout
- `turbo_dir`: path to your local `llama-cpp-turboquant` checkout
- `cuda_root`: CUDA install root used for builds

Example:

```yaml
model_dir: ~/ollama
llama_cpp_dir: ~/Projects/llama.cpp
turbo_dir: ~/Projects/llama-cpp-turboquant
cuda_root: /opt/cuda-13.1
```

These values are used across the CLI:

- `list`, `ctxinfo`, `start`, `run`, and `batch` use `model_dir` to discover local models
- `start`, `run`, and `batch` use `llama_cpp_dir` or `turbo_dir` to launch the selected backend
- `build` uses `llama_cpp_dir`, `turbo_dir`, and `cuda_root`
- `ctxinfo` uses the family metadata in the config to estimate context limits

## Run Models

The main workflow is:

1. pick a model
2. choose a backend
3. start the server
4. run the tests
5. stop the server

Examples:

```bash
# List models discovered in model_dir
test-llama list

# Show which models are currently running
test-llama running

# Show estimated max context for each model
test-llama ctxinfo

# Start a model with llama.cpp defaults
test-llama start qwen27b

# Start the same model on TurboQuant
test-llama start qwen27b --turbo3
test-llama start qwen27b --turbo4

# Start with a custom context size
test-llama start qwen27b --ctx 65536

# Stop a running server
test-llama stop 11435

# Start, test, and stop in one command
test-llama run qwen27b --turbo3
```

## Backends

The two backends we compare are:

- `llama.cpp`
- `turboquant`

The whole point of the repo is to compare them on actual local inference behavior. That means the same model, the same prompt, and the same test harness, so we can see which backend is more useful in practice.

## Benchmarks

There are three different ways to evaluate a model in this repo.

### 1. Task Tests

This is the main benchmark.

```bash
test-llama test 11435
```

This runs the prompt suite in `tests/` against a running model and scores the responses. The current suite focuses on coding and debugging tasks, with the harness checking whether responses actually work.

This is the best tool for comparing `llama.cpp` vs `turboquant`, because it measures usefulness instead of just raw probability metrics.

You can also run it end-to-end with:

```bash
test-llama run qwen27b --turbo3
```

### 2. Context Stress Tests

```bash
test-llama context 11435 file1.txt,file2.txt
```

This is a long-context stress test. It builds prompts that contain a hidden token inside a large text block, then checks whether the model can still recover that token at different context sizes.

This mode is useful for answering questions like:

- does the model stay coherent at long context?
- does it start looping or repeating itself?
- does it still find the important detail buried in the prompt?

### 3. Throughput Bench

```bash
test-llama bench 11435
```

This is a simple speed check for a running server. It measures tokens per second for a synthetic prompt and generation length.

This is useful when you care about throughput, but it is not the main quality benchmark.

## Batch Runs

Batch mode is for comparing many model/backend combinations in one go.

```bash
test-llama batch batch_template.yaml
```

The batch runner:

- starts each model
- runs the task tests
- saves JSON and CSV results in `results/`
- stops the model before moving to the next item

Example batch config:

```yaml
- model: qwen27b
- model: qwen27b
  turbo3: true
- model: Qwen3.6-35B-A3B
  ctk: turbo3
  ctv: turbo3
- model: Qwen3.6-35B-A3B-UD-Q6_K_XL
  fit: true
  no_offload_kv: true
  ctx: 8192
```

## Test Suite

The prompts live in `tests/`.

- `code/` tests coding tasks
- `debugging/` tests bug-fixing tasks

The harness in `test_llama/testing.py` runs those prompts locally against a model server and evaluates the responses. For reproducibility, the test path uses deterministic settings like `temperature=0` and a fixed seed.

## Build

You can build either backend directly from the CLI:

```bash
test-llama build llama.cpp
test-llama build llama.cpp --cpu-only
test-llama build turboquant
test-llama build all --force
```

## Useful Commands

```bash
test-llama tests
test-llama status 11435
test-llama perplexity qwen27b path/to/text.txt
```

`perplexity` is still available, but it is mainly a lower-level diagnostic. The core benchmark story in this repo is the task suite and the context stress suite.
