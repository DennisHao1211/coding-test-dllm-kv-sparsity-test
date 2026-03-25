"""
Standalone speed comparison for baseline and original HF block-cache reuse.

Run:
  cd /home/hice1/hhao40/Fast-dLLM/v2
  python test_speed.py
"""

import csv
import os
import sys
import time
import types
from typing import Optional

import torch

from local_model_loader import load_causal_lm_and_tokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
V2_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(V2_DIR, "local_fast_dllm_1.5B")
DEFAULT_CODE_PATH = os.path.join(V2_DIR, "modified")
MODEL_PATH_OVERRIDE = os.environ.get("FAST_DLLM_MODEL_PATH")
MODEL_PATH = MODEL_PATH_OVERRIDE or DEFAULT_MODEL_PATH
MODEL_CODE_PATH = os.environ.get(
    "FAST_DLLM_CODE_PATH",
    MODEL_PATH if MODEL_PATH_OVERRIDE else (
        DEFAULT_CODE_PATH if os.path.isdir(DEFAULT_CODE_PATH) else MODEL_PATH
    ),
)
MASK_ID    = 151665
STOP_TOKEN = 151645
BD_SIZE    = 32           # large block size (bd_size)
SMALL_BLOCK_SIZE = 16
THRESHOLD        = 0.9
DELAY_STEP       = 1
FIXED_BUDGET_MAX_NEW_TOKENS = 256
TEST_BATCH_SIZE = 1

# GSM8k-style prompts (few samples)
TEST_PROMPTS = [
    (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
        "morning and bakes muffins for her friends every day with four. She sells "
        "the remainder at the farmers' market daily for $2 per fresh duck egg. "
        "How much in dollars does she make every day at the farmers' market?\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber.  "
        "How many bolts in total does it take?\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    ),
    (
        "Josh decides to try flipping a house.  He buys a house for $80,000 and "
        "then puts in $50,000 in repairs.  This increased the value of the house "
        "by 150%.  How much profit did he make?\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    ),
]
TEST_PROMPTS = TEST_PROMPTS[:TEST_BATCH_SIZE]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_path: str, device: str, code_path: str):
    """Load the model weights plus an explicit local modeling implementation."""
    print(f"Loading model weights from {model_path} ...")
    print(f"Loading model code from {code_path} ...")
    model, tokenizer, _ = load_causal_lm_and_tokenizer(
        model_path,
        code_path=code_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()

    import generation_functions  # noqa: E402

    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# NFE instrumentation
# ---------------------------------------------------------------------------
class NFECounter:
    """Wraps model.forward with a call counter to measure NFE."""

    def __init__(self, model):
        self.count = 0
        self._model = model
        self._orig_forward = model.forward

        counter = self

        def _counted_forward(*args, **kwargs):
            counter.count += 1
            return counter._orig_forward(*args, **kwargs)

        model.forward = _counted_forward

    def restore(self):
        self._model.forward = self._orig_forward


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_device_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _effective_stop_token(stop_token: Optional[int]) -> int:
    return -1 if stop_token is None else stop_token


def _extract_generated_tokens(ids, prompt_len: int, pad_token_id: Optional[int],
                              stop_token: Optional[int]):
    gen = ids[prompt_len:]
    gen = gen[gen != MASK_ID]
    if pad_token_id is not None:
        gen = gen[gen != pad_token_id]

    stop_hit = False
    if stop_token is not None:
        stop_positions = (gen == stop_token).nonzero(as_tuple=False)
        if stop_positions.numel() > 0:
            gen = gen[: stop_positions[0].item() + 1]
            stop_hit = True

    return gen, stop_hit


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------
def run_generation(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    *,
    use_block_cache: bool = False,
    use_prefix_filter: bool = True,
    threshold: float = THRESHOLD,
    delay_step: int = DELAY_STEP,
    small_block_size: int = SMALL_BLOCK_SIZE,
    bd_size: int = BD_SIZE,
    max_new_tokens: int = FIXED_BUDGET_MAX_NEW_TOKENS,
    kv_log: Optional[list] = None,
    stop_token: Optional[int] = STOP_TOKEN,
) -> dict:
    """Run batch_sample on prompts and return timing + NFE metrics."""

    encoded = [tokenizer(p, return_tensors="pt")["input_ids"] for p in prompts]
    seq_lens = [e.shape[1] for e in encoded]
    max_len = max(seq_lens)
    min_len = min(seq_lens)

    padded = [
        torch.cat(
            [e, torch.full((1, max_len - e.shape[1]), MASK_ID, dtype=torch.long)],
            dim=1,
        )
        for e in encoded
    ]
    input_ids = torch.cat(padded, dim=0).to(device)
    seq_len_tensor = torch.tensor(seq_lens, device=device)

    nfe_ctr = NFECounter(model)
    try:
        sync_device(device)
        t0 = time.perf_counter()

        finished = model.mdm_sample(
            input_ids,
            tokenizer=tokenizer,
            block_size=bd_size,
            small_block_size=small_block_size,
            max_new_tokens=max_new_tokens,
            mask_id=MASK_ID,
            min_len=min_len,
            seq_len=seq_len_tensor,
            threshold=threshold,
            stop_token=_effective_stop_token(stop_token),
            use_block_cache=use_block_cache,
            use_prefix_filter=use_prefix_filter,
            delay_step=delay_step,
            kv_log=kv_log,
        )

        sync_device(device)
        elapsed = time.perf_counter() - t0
    finally:
        nfe_ctr.restore()

    total_nfe = nfe_ctr.count

    pad_token_id = tokenizer.pad_token_id
    generated_ids = {}
    generated_token_counts = {}
    stop_hits = {}

    total_tokens = 0
    for i, ids in finished.items():
        gen, stop_hit = _extract_generated_tokens(
            ids,
            seq_lens[i],
            pad_token_id,
            stop_token,
        )
        generated_ids[i] = gen
        generated_token_counts[i] = int(gen.shape[0])
        stop_hits[i] = stop_hit
        total_tokens += generated_token_counts[i]

    return {
        "elapsed": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "nfe": total_nfe,
        "nfe_per_sample": total_nfe / len(prompts),
        "seq_lens": seq_lens,
        "generated_ids": generated_ids,
        "generated_token_counts": generated_token_counts,
        "stop_hits": stop_hits,
        "stop_token": stop_token,
    }

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_results(label: str, r: dict, tokenizer, *, show_samples: bool = True) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Elapsed time      : {r['elapsed']:.3f} s")
    print(f"  Tokens generated  : {r['total_tokens']}")
    print(f"  Tokens / second   : {r['tokens_per_sec']:.2f}")
    print(f"  Total NFE         : {r['nfe']}")
    print(f"  NFE / sample      : {r['nfe_per_sample']:.2f}")
    if not show_samples:
        print()
        print("  Sample snippets   : omitted for fixed-budget benchmark")
        print()
        return

    print()
    for i, gen in sorted(r["generated_ids"].items()):
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        snippet = text[:140] + ("..." if len(text) > 140 else "")
        stop_note = ", stop" if r["stop_hits"].get(i) else ""
        print(
            f"  [Sample {i}] tokens={r['generated_token_counts'][i]}{stop_note}  "
            f"{snippet}"
        )
    print()


def print_natural_summary(
    baseline: dict,
    sparse: dict,
    *,
    baseline_label: str = "Baseline",
    candidate_label: str = "Sparse cache",
) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print("  NATURAL GENERATION SUMMARY")
    print(sep)

    b_tps = baseline["tokens_per_sec"]
    s_tps = sparse["tokens_per_sec"]
    throughput_ratio = s_tps / b_tps if b_tps > 0 else float("nan")
    runtime_speedup = baseline["elapsed"] / sparse["elapsed"] if sparse["elapsed"] > 0 else float("nan")

    b_nfe = baseline["nfe"]
    s_nfe = sparse["nfe"]
    nfe_reduction = (1.0 - s_nfe / b_nfe) * 100 if b_nfe > 0 else float("nan")

    print(f"  {'Metric':<30} {baseline_label:>12} {candidate_label:>14}")
    print(f"  {'-'*56}")
    print(f"  {'Generated tokens':<30} {baseline['total_tokens']:>12} {sparse['total_tokens']:>14}")
    print(f"  {'Tokens/sec':<30} {b_tps:>12.2f} {s_tps:>14.2f}")
    print(f"  {'Throughput ratio':<30} {'1.00x':>12} {throughput_ratio:>13.2f}x")
    print(f"  {'Runtime speedup':<30} {'1.00x':>12} {runtime_speedup:>13.2f}x")
    print(f"  {'Total NFE':<30} {b_nfe:>12} {s_nfe:>14}")
    print(
        f"  {'NFE/sample':<30} {baseline['nfe_per_sample']:>12.2f} "
        f"{sparse['nfe_per_sample']:>14.2f}"
    )
    print(f"  {'NFE reduction':<30} {'-':>12} {nfe_reduction:>13.1f}%")
    print(f"  {'Total time (s)':<30} {baseline['elapsed']:>12.3f} {sparse['elapsed']:>14.3f}")
    if baseline["total_tokens"] != sparse["total_tokens"]:
        print()
        print("  NOTE: output lengths differ, so tokens/sec is not a fair speed verdict.")
        print("        Use the fixed-budget benchmark below for apples-to-apples timing.")
    print(sep)


def print_fixed_budget_summary(
    baseline: dict,
    sparse: dict,
    max_new_tokens: int,
    *,
    title: str = "FIXED-BUDGET SPEED SUMMARY",
    baseline_label: str = "Baseline",
    candidate_label: str = "Sparse cache",
) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    runtime_speedup = baseline["elapsed"] / sparse["elapsed"] if sparse["elapsed"] > 0 else float("nan")
    b_nfe = baseline["nfe"]
    s_nfe = sparse["nfe"]
    nfe_reduction = (1.0 - s_nfe / b_nfe) * 100 if b_nfe > 0 else float("nan")
    b_fps = b_nfe / baseline["elapsed"] if baseline["elapsed"] > 0 else 0.0
    s_fps = s_nfe / sparse["elapsed"] if sparse["elapsed"] > 0 else 0.0

    print(f"  Early stop        : disabled")
    print(f"  Max new tokens    : {max_new_tokens} per sample")
    print()
    print(f"  {'Metric':<30} {baseline_label:>12} {candidate_label:>14}")
    print(f"  {'-'*56}")
    print(f"  {'Total time (s)':<30} {baseline['elapsed']:>12.3f} {sparse['elapsed']:>14.3f}")
    print(f"  {'Runtime speedup':<30} {'1.00x':>12} {runtime_speedup:>13.2f}x")
    print(f"  {'Total NFE':<30} {b_nfe:>12} {s_nfe:>14}")
    print(f"  {'Forwards/sec':<30} {b_fps:>12.2f} {s_fps:>14.2f}")
    print(f"  {'NFE reduction':<30} {'-':>12} {nfe_reduction:>13.1f}%")
    print(sep)


# ---------------------------------------------------------------------------
# KV cache CSV logging
# ---------------------------------------------------------------------------
def _simplify_kv_row(row: dict) -> dict:
    """Project verbose KV logging into a compact CSV-friendly schema."""

    current_past_kv = row.get(
        "current_past_kv",
        row.get("past_kv_after", row.get("past_kv", 0)),
    )
    if current_past_kv in ("", None):
        current_past_kv = 0

    past_kv_value = row.get(
        "past_kv_value",
        row.get("past_kv_after", row.get("commit_total_kv", row.get("total_kv", 0))),
    )
    if past_kv_value in ("", None):
        past_kv_value = row.get("total_kv", 0)

    return {
        "block_idx": row.get("block_idx", ""),
        "small_block_idx": row.get("small_block_idx", ""),
        "step": row.get("step", ""),
        "state": row.get("state", ""),
        "cache_state": row.get("cache_state", ""),
        "decode_step": row.get("decode_step", ""),
        "local_mode": row.get("local_mode", ""),
        "dense_prefix_kv": int(row.get("dense_prefix_kv", row.get("past_kv", 0)) or 0),
        "filtered_prefix_kv": int(row.get("filtered_prefix_kv", row.get("past_kv", 0)) or 0),
        "dense_local_kv": int(row.get("dense_local_kv", row.get("context_kv", 0)) or 0),
        "commit_total_kv": int(row.get("commit_total_kv", row.get("total_kv", 0)) or 0),
        "past_kv_value": int(past_kv_value or 0),
        "current_past_kv": int(current_past_kv),
        "unmask_count": row.get("unmask_count", ""),
    }


def write_kv_csv(kv_log: list, filepath: str) -> None:
    """Write a single KV log to CSV."""

    fieldnames = [
        "block_idx",
        "small_block_idx",
        "step",
        "state",
        "cache_state",
        "decode_step",
        "local_mode",
        "dense_prefix_kv",
        "filtered_prefix_kv",
        "dense_local_kv",
        "commit_total_kv",
        "past_kv_value",
        "current_past_kv",
        "unmask_count",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kv_log:
            writer.writerow(_simplify_kv_row(row))

    print(f"  KV log written to: {filepath}  ({len(kv_log)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if V2_DIR not in sys.path:
        sys.path.insert(0, V2_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, device, MODEL_CODE_PATH)

    baseline_kv_log: list = []
    prefix_filter_kv_log: list = []

    total_runs = 2

    print(f"\n[1/{total_runs}] Running BLOCK-CACHE REUSE baseline benchmark ...")
    print(
        f"      early_stop=False, small_block_size={SMALL_BLOCK_SIZE}, "
        f"threshold={THRESHOLD}, delay_step={DELAY_STEP}, use_block_cache=True, use_prefix_filter=False, "
        f"max_new_tokens={FIXED_BUDGET_MAX_NEW_TOKENS}"
    )
    baseline_fixed = run_generation(
        model,
        tokenizer,
        TEST_PROMPTS,
        device,
        use_block_cache=True,
        use_prefix_filter=False,
        threshold=THRESHOLD,
        delay_step=DELAY_STEP,
        small_block_size=SMALL_BLOCK_SIZE,
        bd_size=BD_SIZE,
        max_new_tokens=FIXED_BUDGET_MAX_NEW_TOKENS,
        kv_log=baseline_kv_log,
        stop_token=None,
    )
    print_results(
        "BLOCK-CACHE REUSE BASELINE FIXED-BUDGET  "
        f"(small_block_size={SMALL_BLOCK_SIZE}, "
        f"max_new_tokens={FIXED_BUDGET_MAX_NEW_TOKENS})",
        baseline_fixed,
        tokenizer,
        show_samples=False,
    )

    clear_device_cache(device)

    print(f"[2/{total_runs}] Running PREFIX-FILTER + BLOCK-CACHE benchmark ...")
    print(
        f"      early_stop=False, small_block_size={SMALL_BLOCK_SIZE}, "
        f"threshold={THRESHOLD}, delay_step={DELAY_STEP}, use_block_cache=True, use_prefix_filter=True, "
        f"max_new_tokens={FIXED_BUDGET_MAX_NEW_TOKENS}"
    )
    prefix_filter_fixed = run_generation(
        model,
        tokenizer,
        TEST_PROMPTS,
        device,
        use_block_cache=True,
        use_prefix_filter=True,
        threshold=THRESHOLD,
        delay_step=DELAY_STEP,
        small_block_size=SMALL_BLOCK_SIZE,
        bd_size=BD_SIZE,
        max_new_tokens=FIXED_BUDGET_MAX_NEW_TOKENS,
        kv_log=prefix_filter_kv_log,
        stop_token=None,
    )
    print_results(
        "PREFIX-FILTER + BLOCK-CACHE FIXED-BUDGET  "
        f"(small_block_size={SMALL_BLOCK_SIZE})",
        prefix_filter_fixed,
        tokenizer,
        show_samples=False,
    )

    print_fixed_budget_summary(
        baseline_fixed,
        prefix_filter_fixed,
        max_new_tokens=FIXED_BUDGET_MAX_NEW_TOKENS,
        title="BLOCK-CACHE REUSE VS PREFIX-FILTER SUMMARY",
        candidate_label="Prefix-filter + block-cache reuse",
    )

    baseline_csv_path = os.path.join(V2_DIR, "baseline_kv_log.csv")
    block_cache_csv_path = os.path.join(V2_DIR, "block_cache_reuse_kv_log.csv")
    write_kv_csv(baseline_kv_log, baseline_csv_path)
    write_kv_csv(prefix_filter_kv_log, block_cache_csv_path)
