# Fast-dLLM V2 Coding Test Report

## Papers To Read
- Fastdllm-v1: `https://arxiv.org/abs/2505.22618`
- Fastdllm-v2: `https://arxiv.org/pdf/2509.26328`
- Sparse-dLLM: `https://arxiv.org/pdf/2508.02558`
- dLLM-Cache: `https://arxiv.org/abs/2506.06295`

## Fixed Experimental Setup
- GPU: `NVIDIA H100`
- Batch size: `1`
- Large block size: `32`
- Small block size: `16`
- Delay step: `1`
- Block-cache keep ratio: `0.7`
- Block-cache pool kernel size: `3`
- Max new tokens: `2048`
- Unless otherwise noted, all reported experiments in this document use the setup above.

## Implementation Checklist
- [✅] Read the four papers above.
- [✅] Get familiar with `fastdllm-v2`, `dLLM-Cache`, and `Sparse-dLLM` codebases.
- [✅] Apply Sparse-dLLM-style training-free dynamic cache eviction with sparse attention to `fastdllm-v2-1.5B`.
- [✅] Apply Sparse-dLLM-style training-free dynamic cache eviction with sparse attention to `fastdllm-v2-7B`.
- [✅] Produce main comparison results on GSM8K and MATH.
- [✅] Run delay-step ablation on GSM8K first 100 samples for 1.5B.
- [✅] Run delay-step ablation on GSM8K first 100 samples for 7B.
- [✅] Run retention-ratio ablation on GSM8K first 100 samples for 1.5B.
- [✅] Run retention-ratio ablation on GSM8K first 100 samples for 7B.
- [✅] Sweep delay step and retention ratio across all layers for 1.5B.
- [✅] Sweep delay step and retention ratio across all layers for 7B.
- [ ] Apply dLLM-Cache (V-verify) to `fastdllm-v2-1.5B`.
- [ ] Apply dLLM-Cache (V-verify) to `fastdllm-v2-7B`.
- [ ] Propose a combined Sparse-dLLM + dLLM-Cache method on 1.5B.

## Required Result Tables

### Table A. Main Comparison: Fast-dLLM-V2 vs Sparse-dLLM Variant
Use this for Step 4. Report accuracy, throughput, and peak memory on GSM8K and MATH.

| Model             | Method        | GSM8K Acc (%) | GSM8K TPS | GSM8K Mem (GB) | MATH Acc (%) | MATH TPS | MATH Mem (GB) | Notes                                          |
|-------------------|---------------|--------------:|----------:|---------------:|-------------:|---------:|--------------:|------------------------------------------------|
| Fast-dLLM-v2-1.5B | Baseline      |         58.38 |    150.42 |           3.04 |        28.20 |   191.41 |          3.34 | GSM8K strict-match: 5.38%; MATH stderr: 0.0060 |
| Fast-dLLM-v2-1.5B | + Sparse-dLLM |         59.29 |    141.52 |           3.15 |        28.20 |   179.48 |          3.34 | GSM8K strict-match: 5.84%; MATH stderr: 0.0060 |
| Fast-dLLM-v2-7B   | Baseline      |         79.61 |    179.90 |          14.44 |        45.68 |   181.78 |         14.75 | GSM8K strict-match: 0.23%; MATH stderr: 0.0066 |
| Fast-dLLM-v2-7B   | + Sparse-dLLM |         78.62 |    169.97 |          14.68 |        46.38 |   175.39 |         14.75 | GSM8K strict-match: 0.30%; MATH stderr: 0.0066 |

### Table B. Main Comparison Summary
Use this to summarize the relative change against the baseline.

| Model | Dataset | Metric       | Baseline | Sparse-dLLM | Relative Change |
|-------|---------|--------------|---------:|------------:|----------------:|
| 1.5B  | GSM8K   | Accuracy (%) |    58.38 |       59.29 |           +0.91 |
| 1.5B  | GSM8K   | TPS          |   150.42 |      141.52 |          -5.92% |
| 1.5B  | GSM8K   | Memory (GB)  |     3.04 |        3.15 |          +3.62% |
| 1.5B  | MATH    | Accuracy (%) |    28.20 |       28.20 |           +0.00 |
| 1.5B  | MATH    | TPS          |   191.41 |      179.48 |          -6.23% |
| 1.5B  | MATH    | Memory (GB)  |     3.34 |        3.34 |          +0.00% |
| 7B    | GSM8K   | Accuracy (%) |    79.61 |       78.62 |           -0.99 |
| 7B    | GSM8K   | TPS          |   179.90 |      169.97 |          -5.52% |
| 7B    | GSM8K   | Memory (GB)  |    14.44 |       14.68 |          +1.66% |
| 7B    | MATH    | Accuracy (%) |    45.68 |       46.38 |           +0.70 |
| 7B    | MATH    | TPS          |   181.78 |      175.39 |          -3.51% |
| 7B    | MATH    | Memory (GB)  |    14.75 |       14.75 |          +0.00% |

### Table C. Delay-Step Ablation on GSM8K First 100 Samples
Use this for Step 5. 

#### 1.5B
Baseline (no prefix filter): Accuracy `58.00%`, Throughput `142.43 TPS`, Peak Memory `2.97 GB`, strict-match `6.00%`.
| Delay Step | Accuracy (%) | Throughput (TPS) | Peak Memory (GB) | Notes               |
|-----------:|-------------:|-----------------:|-----------------:|---------------------|
|          0 |        59.00 |           143.43 |             3.16 | strict-match: 4.00% |
|          1 |        63.00 |           137.05 |             2.99 | strict-match: 6.00% |
|          2 |        61.00 |           146.68 |             3.06 | strict-match: 6.00% |
|          3 |        61.00 |           148.18 |             3.06 | strict-match: 6.00% |
|          4 |        63.00 |           150.27 |             3.06 | strict-match: 5.00% |
|          5 |        56.00 |           143.75 |             2.99 | strict-match: 5.00% |

#### 7B
Baseline (no prefix filter): Accuracy `73.00%`, Throughput `178.34 TPS`, Peak Memory `14.34 GB`, stderr `0.0446`.
| Delay Step | Accuracy (%) | Throughput (TPS) | Peak Memory (GB) | Notes               |
|-----------:|-------------:|-----------------:|-----------------:|---------------------|
|          0 |        75.00 |           167.44 |            14.41 | stderr: 0.0435      |
|          1 |        71.00 |           172.42 |            14.68 | stderr: 0.0456      |
|          2 |        78.00 |           178.06 |            14.52 | stderr: 0.0416      |
|          3 |        77.00 |           172.73 |            14.40 | stderr: 0.0423      |
|          4 |        78.00 |           175.80 |            14.39 | stderr: 0.0416      |
|          5 |        76.00 |           177.75 |            14.38 | stderr: 0.0429      |

### Table D. Retention-Ratio Ablation on GSM8K First 100 Samples
Use this for Step 6.

#### 1.5B
Baseline (no prefix filter): Accuracy `58.00%`, Throughput `147.51 TPS`, Peak Memory `2.97 GB`, strict-match `6.00%`.
| Retention Ratio | Accuracy (%) | Throughput (TPS) | Peak Memory (GB) | Notes               |
|----------------:|-------------:|-----------------:|-----------------:|---------------------|
|             0.1 |         6.00 |           162.65 |             3.05 | strict-match: 0.00% |
|             0.2 |        17.00 |           137.95 |             3.08 | strict-match: 1.00% |
|             0.3 |        40.00 |           140.55 |             3.09 | strict-match: 4.00% |
|             0.4 |        42.00 |           148.22 |             3.09 | strict-match: 8.00% |
|             0.5 |        55.00 |           154.43 |             3.12 | strict-match: 5.00% |
|             0.6 |        56.00 |           141.97 |             2.99 | strict-match: 6.00% |
|             0.7 |        63.00 |           144.27 |             2.99 | strict-match: 6.00% |
|             0.8 |        63.00 |           143.34 |             3.00 | strict-match: 8.00% |
|             0.9 |        59.00 |           148.66 |             3.11 | strict-match: 6.00% |
|             1.0 |        60.00 |           146.13 |             2.99 | strict-match: 7.00% |

#### 7B
Baseline (no prefix filter): Accuracy `73.00%`, Throughput `178.54 TPS`, Peak Memory `14.34 GB`, stderr `0.0446`.
| Retention Ratio | Accuracy (%) | Throughput (TPS) | Peak Memory (GB) | Notes               |
|----------------:|-------------:|-----------------:|-----------------:|---------------------|
|             0.1 |        20.00 |           165.32 |            14.47 | stderr: 0.0402      |
|             0.2 |        45.00 |           155.42 |            14.50 | stderr: 0.0500      |
|             0.3 |        64.00 |           158.56 |            14.36 | stderr: 0.0482      |
|             0.4 |        71.00 |           161.49 |            14.37 | stderr: 0.0456      |
|             0.5 |        78.00 |           167.20 |            14.37 | stderr: 0.0416      |
|             0.6 |        79.00 |           164.46 |            14.38 | stderr: 0.0409      |
|             0.7 |        71.00 |           170.36 |            14.68 | stderr: 0.0456      |
|             0.8 |        75.00 |           167.96 |            14.40 | stderr: 0.0435      |
|             0.9 |        76.00 |           165.71 |            14.41 | stderr: 0.0429      |
|             1.0 |        77.00 |           171.43 |            14.39 | stderr: 0.0423      |

### Table E. Joint Sweep: Delay Step x Retention Ratio
Use this for Step 7. 

#### 1.5B
| Delay Step | Retention Ratio | Accuracy (%) | Throughput (TPS) | Peak Memory (GB) | Notes               |
|-----------:|----------------:|-------------:|-----------------:|-----------------:|---------------------|
|          3 |             0.5 |        54.00 |           150.56 |             3.07 | strict-match: 6.00% |
|          4 |             0.5 |        54.00 |           153.90 |             3.05 | strict-match: 5.00% |
|          4 |             0.7 |        63.00 |           155.67 |             3.06 | strict-match: 5.00% |

#### 7B
| Delay Step | Retention Ratio | Accuracy (%) | Throughput (TPS) | Peak Memory (GB) | Notes               |
|-----------:|----------------:|-------------:|-----------------:|-----------------:|---------------------|
|          2 |             0.5 |        79.00 |           166.07 |            14.38 | stderr: 0.0409      |
|          2 |             0.6 |        79.00 |           168.76 |            14.39 | stderr: 0.0409      |
|          2 |             0.7 |        78.00 |           177.36 |            14.52 | stderr: 0.0416      |

### Table F. dLLM-Cache (V-verify) Results
Use this for Step 8.

| Model | Dataset | Method | Accuracy (%) | Throughput (TPS) | Peak Memory (GB) | Notes |
|---|---|---|---:|---:|---:|---|
| 1.5B | GSM8K-100 | dLLM-Cache (V-verify) |  |  |  | |
| 1.5B | MATH-100 | dLLM-Cache (V-verify) |  |  |  | |
| 7B | GSM8K-100 | dLLM-Cache (V-verify) |  |  |  | |
| 7B | MATH-100 | dLLM-Cache (V-verify) |  |  |  | |

### Table G. Combined Sparse-dLLM + dLLM-Cache Method
Use this for Step 9.

| Model | Method | Dataset | Accuracy (%) | Throughput (TPS) | Peak Memory (GB) | Notes |
|---|---|---|---:|---:|---:|---|
| 1.5B | Sparse-dLLM Baseline | GSM8K-100 |  |  |  | |
| 1.5B | Sparse+dLLM-Cache Proposed | GSM8K-100 |  |  |  | |

## Deliverables Checklist

### 1. Code Deliverable
- [✅] Code implementing Sparse-dLLM techniques on Fast-dLLM-v2.
- [ ] Code implementing dLLM-Cache on Fast-dLLM-v2.
- [✅] Code or scripts used to generate the requested experiment tables/plots.

### 2. Report Deliverable
Answer the following in the final report.

- [✅] What are the differences between Fast-dLLM-v2 and LLaDA as used in Sparse-dLLM?
- [✅] What techniques are proposed in Sparse-dLLM? Do they make sense? Why or why not?
- [✅] What techniques are proposed in dLLM-Cache? Do they make sense? Why or why not?
- [✅] Results with and without Sparse-dLLM techniques in terms of accuracy, throughput, and memory on GSM8K and MATH, with summarized observations.
- [✅] Ablation study on the delay step for GSM8K, with summarized observations.
- [✅] Ablation study on the retention ratio for GSM8K, with summarized observations.
- [ ] Step 8 dLLM-Cache results.
- [ ] Step 9 combined-method results.

## Report Questions To Answer

### 1. What are the differences between Fast-dLLM-v2 and LLaDA as used in Sparse-dLLM?
Fast-dLLM-v2 is an AR-style block decoding workflow with committed history KV plus current local block KV (both small blocks and blocks are generated one by one), while LLaDA in Sparse-dLLM is iterative diffusion denoising over masked tokens per block. Fast-dLLM-v2 uses past_key_values (global committed prefix) + block_past_key_values, while LLaDA uses per-block CustomCache and MDM path does not rely on standard autoregressive past_key_values caching. In Fast-dLLM-v2, sparse prefix filtering is happened outside the layer forward via _build_layerwise_prefix_sparse_cache. Why? Since in Fast-dLLM-v2, Sparse-dLLM is scheduled at the block level: building the sparse prefix once from the current block’s query states, then reuse it throughout that block’s reuse phase. It is still layerwise; only the execution location is lifted out of the layer forward. Instead of filtering inside each layer’s forward pass, first collect per-layer Q_b, then iterate by layer_idx in _build_layerwise_prefix_sparse_cache to compute per-layer importance/top-k and produce per-layer filtered KV. In LLaDA, filtering is done inside attention forward via customcache.filter_cache(...) when cache_state==1.

### 2. What techniques are proposed in Sparse-dLLM? Do they make sense? Why or why not?
Training-free dynamic cache eviction (Prefix-Sparse): For each layer, score historical KV tokens using current-block query signals, then keep only top-r important tokens (with optional max-pooling smoothing). Delayed cache update schedule: Use a warmup/build/reuse schedule so sparse filtering is built from more stable queries before repeated reuse. Sparse-dLLM is more natural for LLaDA/Dream, because one block-level sparse cache can be reused across many denoising steps, so filter cost is well amortized. In Fast-dLLM-v2’s hybrid workflow (large block split into many small blocks), each new small block tends to trigger a new build/filter cycle, so you repeatedly pay for query collection + layerwise scoring + top-k gather/scatter. That creates a stronger tradeoff: Benefit: shorter prefix context during reuse (lower attention cost). Cost: repeated filtering overhead can cancel the speed gain, especially when small blocks has smaller size, (more filter states in the big block and less reuse states per small block). Quality risk: filtering from partially updated/noisier queries can be unstable, and aggressive pruning may drop long-range evidence (notably harmful for reasoning-heavy tasks). So the method is valid, but the cost-benefit profile is less favorable in v2 unless filtering frequency is reduced (e.g., one filter per large block, not per small block: I tried this, but it is unrealistic because early in a large block most tokens are still masked/uncertain, so the query-based importance scores are not representative for later small-block decoding. A single filter quickly becomes stale and hurts accuracy; recomputing it later for better quality needs extra dense passes and more forwards (the number of tokens being unmasked becomes fewer for each reuse step), which cancels much of the speed gain.) Also, Fast-dLLM-v2 does not have suffix context in this hybrid decoding path, so Algorithm 1 (Dynamic Bidirectional Cache Eviction) cannot be directly applied. As a result, I can only use the compromise variant, Algorithm 5 (Prefix-Sparse), which sparsifies historical prefix KV while keeping local block KV dense.

### 3. What techniques are proposed in dLLM-Cache? Do they make sense? Why or why not?
dLLM-Cache proposes a differentiated caching framework for diffusion-based large language models by exploiting two kinds of redundancy during inference: the quasi-static nature of prompt tokens and the sparse, uneven evolution of response tokens across denoising steps. For the prompt side, it introduces Long-Interval Prompt Caching, which stores prompt-related features such as K, V, attention outputs, and FFN outputs, and refreshes them only at relatively long intervals instead of recomputing them at every denoising step. For the response side, it introduces Adaptive Short-Interval Response Caching, where response features are refreshed more frequently, but not always fully recomputed. Instead, the method uses V-verify, a lightweight mechanism that measures the cosine similarity between a response token’s current and cached Value vectors to identify which tokens have changed the most. Only those low-similarity tokens are selectively recomputed, while the rest reuse cached features. Together, these techniques aim to reduce repeated computation while preserving output quality. These techniques make sense depends strongly on the model architecture. For Fast-dLLM v2, the ideas are only partially compatible rather than directly transferable. The prompt-side intuition still makes sense: if the prompt remains static across denoising steps, then caching prompt-related features should still be useful in principle. But the response-side mechanism is much less straightforward for Fast-dLLM v2 because its inference procedure is block-based and uses more complicated cache and update behavior than the LLaDA-style setting assumed in dLLM-Cache. In particular, Fast-dLLM v2 performs generation and unmasking at the level of blocks and sub-blocks, and its custom cache/state transitions introduce additional forward-attention passes and storage overhead. That means even if only a subset of response tokens is selected for recomputation, the system may still need to pay substantial structural overhead to maintain consistency across the blockwise decoding process. As a result, the theoretical savings from selective token updates may be offset—or even outweighed—by the extra cost of managing these block-level dependencies.

### 4. Results with and without Sparse-dLLM techniques in terms of accuracy, throughput, and memory on GSM8K and MATH, along with summarized observations
With Sparse-dLLM enabled, accuracy is mostly preserved and sometimes improved, but throughput consistently drops while memory changes are small. This is largely because of the tradeoff between the cost of filter state and the gain from the reuse state.  For 1.5B: On GSM8K, accuracy increases from 58.38% to 59.29% (+0.91), but TPS drops from 150.42 to 141.52 (-5.92%) and memory rises from 3.04 GB to 3.15 GB (+3.62%). On MATH, accuracy stays at 28.20% (no change), TPS drops from 191.41 to 179.48 (-6.23%), and memory stays 3.34 GB. For 7B: On GSM8K, accuracy decreases slightly from 79.61% to 78.62% (-0.99), TPS drops from 179.90 to 169.97 (-5.52%), and memory increases from 14.44 GB to 14.68 GB (+1.66%). On MATH, accuracy improves from 45.68% to 46.38% (+0.70), TPS drops from 181.78 to 175.39 (-3.51%), and memory is unchanged (14.75 GB). Overall, this Sparse-dLLM adaptation gives a mixed but generally stable accuracy profile, while showing a consistent speed penalty in this Fast-dLLM-v2 workflow; memory impact is minor.

### 5. Ablation study on the delay step for the GSM8K dataset, with summarized observations
Delay step has a non-monotonic effect, showing a clear quality-speed tradeoff. For 1.5B (GSM8K-100), the best accuracy appears at delay=1/4 (63%), while the best throughput appears near delay=4 (150.27 TPS). Very small delay can make filtering less stable, and too large delay (e.g., 5) hurts accuracy (56%), likely because sparse reuse starts too late to be consistently beneficial. For 7B (GSM8K-100), best accuracy is at delay=2/4 (78%), and throughput is strongest around delay=2 (178.06 TPS) or 5 (177.75 TPS). This suggests moderate delay gives better query maturity before sparse cache build; overly early or mismatched delay degrades quality (e.g., 71% at delay=1). Overall, mid-range delay steps are the most robust for this hybrid Fast-dLLM-v2 workflow: they usually preserve or improve accuracy while keeping throughput close to baseline.

### 6. Ablation study on the retention ratio for the GSM8K dataset, with summarized observations
Retention ratio is the dominant quality knob, with very low ratios causing severe accuracy collapse. For 1.5B (GSM8K-100), accuracy rises sharply from 6% (r=0.1) to 63% (r=0.7/0.8), then slightly drops at r=0.9 (59%) and r=1.0 (60%). This indicates aggressive prefix pruning removes essential evidence, while moderate-high retention gives the best quality. Throughput does not improve monotonically; the best speed appears around r=0.1 (162.65 TPS) but with unacceptable accuracy, so it is not a practical operating point. For 7B (GSM8K-100), the same pattern holds: 20% (r=0.1) to 79% (r=0.6), with strong quality also at r=0.5/0.7/1.0. Mid-to-high retention is consistently safer for accuracy, while speed differences across ratios are comparatively modest and noisy. Overall, the ablation shows a clear tradeoff: low retention favors sparsity but harms correctness; mid-high retention (about 0.5–0.8) is the practical range for maintaining quality in this Fast-dLLM-v2 setting.

### 7. Report Step 8 results
I attempted to apply dLLM-Cache to Fast-dLLM-v2, but it did not improve speed; throughput dropped significantly. The main issue is overhead: dLLM-Cache stores and updates four intermediate features per layer (K, V, AttnOut, FFNOut), so cache cost scales tremendously. In practice, this introduces heavy memory traffic (extra reads/writes/scatter updates and refresh-index computation), which can dominate runtime. In v2’s large-block + small-block workflow, these cache-maintenance costs are paid frequently, while the baseline dual-cache block reuse is already very efficient. As a result, the saved attention compute is not enough to offset the added cache overhead. dLLM-Cache is generally more favorable for very long prompts and long generations (where reuse is better amortized), but for GSM8K/MATH under this v2 setup, achieving net speedup is difficult.

### 8. Report Step 9 results

## Suggested Artifact Bundle
- `modified/modified_modeling.py`
- `generation_functions.py`
- `eval.py`
- `test_speed.py`
- Benchmark logs / CSVs
- Final report PDF or Markdown

## Notes
After studying Sparse-dLLM and tracing the Fast-dLLM-v2 architecture, my initial plan was to reuse the Sparse-dLLM implementation pattern from Dream/LLaDA as much as possible. Based on the paper’s pseudocode, I first assumed sparsification should be inserted where diffusion-style parallel decoding happens. However, in v2, because of its unmasking mechanism, diffusion-like behavior only exists inside each small block (multi-step denoising over unmasked tokens). I then tried adding a new CustomCache in modeling, but that required maintaining three KV caches at once: global historical past_key_values, block-local block_past_key_values, and CustomCache. This made the workflow much more complex, increased scheduling overhead in generation_functions, and largely broke the original v2 block-reuse design. I realized that v2’s native block-reuse cache already plays a role similar to Sparse-dLLM’s custom cache. Simply filtering during small-block reuse of block_past_key_values did not improve speed, because filtering only the current large block was insufficient; following the paper’s intent, the filtering target should be the full historical sequence excluding the current block. Therefore, I finalized the current design: for each small block, use delay-step states 0/1/2; in the build stage, collect per-layer Q_b and apply layerwise prefix filtering via _build_layerwise_prefix_sparse_cache; in the reuse stage, run forward with filtered_past_key_values (sparse historical prefix) plus block_past_key_values (dense local block). block_past_key_values is updated in place at every small-block/forward iteration, and after the full large block is finished, the newly unmasked 32 tokens are committed to past_key_values. I believe the main reason my Sparse-dLLM implementation did not deliver a significant throughput gain over the baseline is that the baseline’s dual-cache block-reuse mechanism is already highly efficient. In the baseline, block reuse is enabled once block_past_key_values has been built and the current small-block start position is no longer masked; otherwise it falls back to full-attention updates. This strong built-in reuse path leaves limited room for additional speedup from prefix filtering, so the overhead introduced by the filter/build states can outweigh the compute savings gained during reuse. At the same time, the large-block-then-small-block workflow makes it difficult to implement the paper’s Sparse-dLLM filtering mechanism in an equivalent way. Scoring KV against the corresponding query states becomes more complicated under this hierarchical schedule, and the implementation is more error-prone.

For the dLLM-Cache, my implementation plan: 
    1. Add a runtime cache manager module, based on the Cache.py and the Selective update pattern in cache_hook_Dream.py and cache_hook_LLaDA.py
        C_p[layer]: kv_cache (prompt K/V only in v2-equivalent mode). No AttnOut_p/FFNOut_p because unlike other diffusion models, for the Fast-dLLM-v2 AR/block decoding pipeline, the prompt acts as historical prefix context and is mainly consumed through prefix KV in attention. Reusing AttnOut_p/FFNOut_p is not useful at all
        C_r[layer]: kv_cache, attn, mlp for response tokens.
        step counter, Kp, Kr, rho, refresh flags, block id, response length.
        refresh_index (cosine similarity, lowest-similarity top-k)
        gather/scatter utilities for selective updates
    2. Write dLLM-cache args from eval to generation
        use_dllm_cache, prompt_interval_steps, gen_interval_steps, transfer_ratio, verify_mode, similarity_metric
    3. Integrate algorithm cases into modified_modeling.py
        Add dLLM-cache kwargs pass-through in CausalLM -> Model -> DecoderLayer -> Attention
        Implement case behavior in reuse path:
        Case 1 (full refresh): recompute response features and overwrite C_r.
        Case 2 (refresh prompt only): sync C_p from prefix cache, reuse C_r response features.
        Case 3 (refresh response only): recompute response features with current prompt KV + response KV.
        Case 4 (adaptive): recompute selected response tokens by V-verify and scatter-update C_r.kv/attn/mlp.
