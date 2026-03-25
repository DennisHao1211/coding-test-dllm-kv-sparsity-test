import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.utils import auto_docstring


# Constants for Fast_dLLM model
FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645

MASK_COLOR = 0.5
TOKEN_COLOR = -0.5


def _cache_seq_len(cache) -> int:
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except Exception:
        pass
    try:
        return int(cache.key_cache[0].shape[2])
    except Exception:
        return 0


def _past_kv_size(past_key_values) -> int:
    """Number of cached prefix tokens in layer 0."""
    return _cache_seq_len(past_key_values)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _gather_kv_per_sample(key_states, value_states, keep_indices):
    gather_index = keep_indices[:, None, :, None].expand(
        -1,
        key_states.shape[1],
        -1,
        key_states.shape[3],
    )
    filtered_k = torch.gather(key_states, dim=2, index=gather_index)
    filtered_v = torch.gather(value_states, dim=2, index=gather_index)
    return filtered_k, filtered_v


def _build_layerwise_prefix_sparse_cache(
    query_state_collector,
    past_key_values,
    *,
    keep_ratio,
    pool_kernel_size,
):
    """
    Build a temporary Sparse-dLLM-style prefix cache for the current decode block.

    Inputs:
    - `query_state_collector`: per-layer query states Q_b captured from one full-block
      forward pass over the current block.
    - `past_key_values`: the dense committed history cache. In this AR adaptation it only
      contains historical prefix tokens that have already been fully decoded and committed.

    What this function does:
    1. For each layer, score every historical prefix KV position using the current block's
       query states.
    2. Keep only the top-k prefix positions according to `keep_ratio`.
    3. Return a new `DynamicCache` that contains the filtered historical prefix only.

    What this function does NOT do:
    - It does not touch the current block's local KV. The local block stays dense and is
      handled separately by `block_past_key_values`.
    - It does not build a suffix cache. Unlike the original diffusion setting, this AR
      adaptation only has historical prefix context available.
    """
    if past_key_values is None or _cache_seq_len(past_key_values) == 0:
        return None

    filtered_cache = DynamicCache()
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))

    for layer_idx in range(len(query_state_collector)):
        q_block = query_state_collector[layer_idx]
        if q_block is None:
            raise ValueError(f"Missing query states for layer {layer_idx}")
        if len(past_key_values) <= layer_idx:
            continue

        key_states, value_states = past_key_values[layer_idx]
        prefix_len = key_states.shape[2]
        if prefix_len == 0:
            continue

        q_heads = q_block.shape[1]
        kv_heads = key_states.shape[1]
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"Incompatible attention heads for sparse cache build: q_heads={q_heads}, kv_heads={kv_heads}"
            )

        # 这里按 Sparse-dLLM / Algorithm 5 的思路，只对“历史 prefix”做重要性估计。
        # 由于 Fast-dLLM v2 这里是 AR 生成，当前时刻只有历史上下文，没有 diffusion
        # setting 里的 future suffix，因此这里不会再额外拼一个 suffix cache。
        #
        # q_block: [batch, q_heads, block_tokens, head_dim]
        # key_states: [batch, kv_heads, prefix_len, head_dim]
        #
        # 先把当前 block 的 query 在 token 维度上做平均，得到每个 head 的块级查询
        # 表示 q_bar。这样得到的是“当前 block 整体最关心什么历史信息”。
        q_bar = q_block.mean(dim=2)

        # Multi-query / grouped-query attention 下，query head 数可能大于 kv head 数。
        # 为了做逐 head 打分，需要先把 K 扩展到和 Q 同样的 head 数。
        expanded_keys = _repeat_kv(key_states, q_heads // kv_heads)

        # 对每个历史 prefix token 计算重要性：
        # importance[b, t] ~ mean_h <q_bar[b, h], K_prefix[b, h, t]>
        # 分数越高，说明当前 block 越依赖这个历史位置。
        importance = torch.matmul(q_bar.unsqueeze(-2), expanded_keys.transpose(-2, -1)).squeeze(-2)
        importance = importance.mean(dim=1)

        # 可选的 1D max-pooling 对应 Sparse-dLLM 里常见的局部平滑思想：
        # 某个位置附近只要存在非常重要的 token，就让邻域整体更容易被保留，
        # 避免因为单点噪声导致前缀裁剪过于尖锐。
        if pool_kernel_size is not None and pool_kernel_size > 1:
            importance = F.max_pool1d(
                importance.unsqueeze(1),
                kernel_size=pool_kernel_size,
                stride=1,
                padding=pool_kernel_size // 2,
            ).squeeze(1)[..., :prefix_len]

        # keep_ratio 决定 prefix 里保留多少历史位置。
        # 至少保留 1 个 token，避免把历史上下文完全裁空。
        keep_num = max(1, int(prefix_len * keep_ratio))
        keep_num = min(prefix_len, keep_num)

        if keep_num >= prefix_len:
            filtered_k = key_states
            filtered_v = value_states
        else:
            # 先按重要性取 top-k，再排序恢复原始时序。
            # 排序很重要：attention 仍然需要按照时间顺序接收 KV。
            keep_indices = torch.topk(importance, k=keep_num, dim=-1).indices
            keep_indices = torch.sort(keep_indices, dim=-1).values
            filtered_k, filtered_v = _gather_kv_per_sample(key_states, value_states, keep_indices)

        # 这个 filtered_cache 只保存“稀疏后的历史 prefix KV”。
        # 当前 block 的 local dense KV 不在这里，后面会通过 block cache 单独拼接。
        filtered_cache.update(filtered_k, filtered_v, layer_idx, cache_kwargs={})

    return filtered_cache if len(filtered_cache) > 0 else None


def _append_kv_log(
    kv_log,
    block_idx,
    small_block_idx,
    step,
    state,
    past_kv,
    context_kv,
    total_kv,
    **extra,
):
    row = {
        "block_idx": block_idx,
        "small_block_idx": small_block_idx,
        "step": step,
        "state": state,
        "past_kv": past_kv,
        "context_kv": context_kv,
        "total_kv": total_kv,
    }
    row.update(extra)
    kv_log.append(row)
    return row


def _append_compact_kv_log(
    kv_log,
    block_idx,
    small_block_idx,
    step,
    state,
    current_past_kv,
    context_kv,
    total_kv,
    *,
    local_mode="",
    dense_prefix_kv=0,
    filtered_prefix_kv=0,
    dense_local_kv=0,
    commit_total_kv=0,
    past_kv_value="",
    **extra,
):
    return _append_kv_log(
        kv_log,
        block_idx,
        small_block_idx,
        step,
        state,
        current_past_kv,
        context_kv,
        total_kv,
        local_mode=local_mode,
        dense_prefix_kv=dense_prefix_kv,
        filtered_prefix_kv=filtered_prefix_kv,
        dense_local_kv=dense_local_kv,
        commit_total_kv=commit_total_kv,
        past_kv_value=past_kv_value,
        **extra,
    )


def _trim_finished_rows(
    x_t,
    *,
    finished_flag,
    seq_len,
    block_idx,
    block_size,
    stop_token,
    pad_token_id,
):
    """Pad out tokens after the first emitted stop token for finished rows."""
    if stop_token is None or stop_token < 0:
        return

    block_limit = (block_idx + 1) * block_size
    for sample_idx in range(x_t.shape[0]):
        if not finished_flag[sample_idx]:
            continue

        seq_start = int(seq_len[sample_idx].item())
        if seq_start >= block_limit:
            continue

        stop_positions = (x_t[sample_idx, seq_start:] == stop_token).nonzero(as_tuple=False)
        if stop_positions.numel() == 0:
            continue

        stop_offset = int(stop_positions[0].item())
        x_t[sample_idx, seq_start + stop_offset + 1 :] = pad_token_id


@auto_docstring
class Fast_dLLM_QwenForCausalLM:

    @torch.no_grad()
    def batch_sample(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=FAST_DLLM_MASK_ID,
        threshold=0.95,
        stop_token=FAST_DLLM_STOP_TOKEN,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
        kv_log=None,
        block_cache_keep_ratio=0.7,
        block_cache_pool_kernel_size=3,
        use_prefix_filter=True,
        delay_step=0,
    ):
        # 这条解码路径是当前仓库里真正承载 Sparse-dLLM 机制的主逻辑。
        #
        # 三个核心状态：
        # 1. `past_key_values`
        #    已经完整提交(commit)的历史 block，对应全局 dense prefix cache。
        # 2. `filtered_past_key_values`
        #    基于当前 block 的 query，对 `past_key_values` 做 top-k 裁剪后得到的
        #    临时 sparse prefix cache。它只在“当前 block 的 reuse 阶段”生效。
        # 3. `block_past_key_values`
        #    当前 block 的 dense local KV cache。即使 prefix 被裁剪，local block 仍然
        #    保持 dense，这也是这版 AR 适配尽量稳住准确率的关键。
        #
        # 当 use_prefix_filter=False 时，这条路径退化回 Fast-dLLM v2 原始的
        # block-cache reuse baseline。
        #
        # `delay_step` 控制 delayed update：
        # - 早期 decode step 先 dense warmup
        # - 到达 delay_step 时 build sparse/dense cache
        # - 之后 reuse cache
        #
        # 这样做的动机是：等当前 small block 里先露出一部分 token，再依据更稳定的
        # query 分布去筛 prefix，会比一开始就裁剪 prefix 更稳。

        delay_step = int(delay_step)
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]

        if min_len > block_size:
            # 先把所有样本共享的、完整落在 block 边界上的 prompt 前缀一次性 prefill 到
            # dense global prefix cache 里。这样后面进入 block-wise decoding 时，
            # `past_key_values` 已经包含了可复用的历史前缀。
            prefix_len = (min_len // block_size) * block_size
            output = self.forward(
                input_ids=input_ids[:, :prefix_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = seq_len == min_len
                if predict_sample_idx.any():
                    predict_logits = logits[predict_sample_idx, -1:, :]
                    next_token = predict_logits.argmax(dim=-1)
                    if input_ids.shape[1] <= min_len:
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                    else:
                        input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        for block_idx in range(start_block_idx, num_blocks):
            if sample_indices.numel() == 0 or finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                # 当前 block 还没显式展开出来时，把它初始化成全 mask。
                block_pad = block_size - input_ids.shape[1] % block_size
                x_init = mask_id * torch.ones(
                    (input_ids.shape[0], block_pad),
                    device=self.device,
                    dtype=torch.long,
                )
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1) * block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            # block_past_key_values:
            #   当前 block 的 dense local KV。只服务于这个 block 内的 small-block reuse。
            block_past_key_values = None
            # filtered_past_key_values:
            #   基于当前 block 的 query，从历史 dense prefix 中筛出来的 sparse prefix。
            #   每个 block build 一次，commit 后失效。
            filtered_past_key_values = None
            block_log_step = 0

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    # 当前 block 已经没有 mask，说明这一整块解完了。
                    # 这时把整块 dense 地 commit 回 `past_key_values`，使它成为之后 block 的
                    # 历史 prefix。一旦 commit 完，本 block 的临时 sparse cache / local
                    # cache 就都可以丢掉了。
                    _trim_finished_rows(
                        x_t,
                        finished_flag=finished_flag,
                        seq_len=seq_len,
                        block_idx=block_idx,
                        block_size=block_size,
                        stop_token=stop_token,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    if finished_flag.all():
                        break

                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    filtered_past_key_values = None
                    block_past_key_values = None
                    if kv_log is not None:
                        commit_total_kv = _past_kv_size(past_key_values)
                        prefix_kv = max(commit_total_kv - block_size, 0)
                        _append_compact_kv_log(
                            kv_log,
                            block_idx,
                            "",
                            block_log_step,
                            "commit",
                            prefix_kv,
                            block_size,
                            commit_total_kv,
                            local_mode="commit_block",
                            dense_prefix_kv=prefix_kv,
                            filtered_prefix_kv=prefix_kv,
                            dense_local_kv=block_size,
                            commit_total_kv=commit_total_kv,
                            past_kv_value=commit_total_kv,
                            cache_state="",
                            decode_step="",
                        )
                    block_log_step += 1

                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size
                    # decode_step 统计“当前 small block 已经迭代了几轮”。
                    # 它不是全局时间步，而是一个局部调度器，用来控制 delay_step 触发。
                    decode_step = 0

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        current_past_kv = _past_kv_size(past_key_values)
                        log_row = None
                        cache_state = 0

                        if use_block_cache:
                            # Delayed cache update:
                            # 0 = dense warmup
                            #     还不建立 sparse prefix / local cache，直接对整个 block 做
                            #     一次 dense forward，先让当前 small block 露出一些 token。
                            # 1 = build cache
                            #     用当前 block 的完整 forward 结果建立：
                            #       - dense local block cache
                            #       - optional sparse prefix cache
                            # 2 = reuse cache
                            #     后续只对当前 small block 的 token 做前向，同时复用：
                            #       - sparse historical prefix
                            #       - dense current local block
                            if decode_step < delay_step:
                                cache_state = 0
                            elif decode_step == delay_step:
                                cache_state = 1
                            else:
                                cache_state = 2

                            if cache_state == 0:
                                # Dense warmup 阶段完全不裁剪 prefix，也不建立 block cache。
                                # 这里的目标不是省算，而是先拿到更稳定的 partial prediction，
                                # 再在后续 build 阶段决定该保留哪些历史 prefix 位置。
                                output = self.forward(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                )
                                logits = output.logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                                if kv_log is not None:
                                    log_row = _append_compact_kv_log(
                                        kv_log,
                                        block_idx,
                                        small_block_idx,
                                        block_log_step,
                                        "dense_no_cache",
                                        current_past_kv,
                                        block_size,
                                        current_past_kv + block_size,
                                        local_mode="dense_no_cache",
                                        dense_prefix_kv=current_past_kv,
                                        filtered_prefix_kv=current_past_kv,
                                        dense_local_kv=block_size,
                                        commit_total_kv=current_past_kv + block_size,
                                        cache_state=cache_state,
                                        decode_step=decode_step,
                                    )
                            elif cache_state == 1:
                                # Full-block build 阶段：
                                # 1. 对整个当前 block 跑一次 forward；
                                # 2. 抓取每层 query states 作为当前 block 的 Q_b；
                                # 3. 建立当前 block 的 dense local KV cache；
                                # 4. 仅对历史 prefix 做 top-k 裁剪，生成 sparse prefix cache。
                                #
                                # 注意：这里“历史 prefix”和“当前 block”是分开的：
                                # - prefix 稀疏
                                # - local block 仍然 dense
                                if use_prefix_filter:
                                    query_state_collector = [None] * self.config.num_hidden_layers
                                    output = self.forward(
                                        input_ids=x_t[:, -block_size:],
                                        use_cache=True,
                                        past_key_values=past_key_values,
                                        update_past_key_values=False,
                                        use_block_cache=True,
                                        record_block_queries=True,
                                        query_state_collector=query_state_collector,
                                    )
                                    filtered_past_key_values = _build_layerwise_prefix_sparse_cache(
                                        query_state_collector,
                                        past_key_values,
                                        keep_ratio=block_cache_keep_ratio,
                                        pool_kernel_size=block_cache_pool_kernel_size,
                                    )
                                else:
                                    output = self.forward(
                                        input_ids=x_t[:, -block_size:],
                                        use_cache=True,
                                        past_key_values=past_key_values,
                                        update_past_key_values=False,
                                        use_block_cache=True,
                                    )
                                    filtered_past_key_values = None

                                logits = output.logits
                                block_past_key_values = output.block_past_key_values
                                # dense_local_kv 表示当前 block dense local cache 的长度；
                                # filtered_prefix_kv 表示裁剪后的 prefix 长度。
                                dense_local_kv = _cache_seq_len(block_past_key_values)
                                filtered_prefix_kv = (
                                    _cache_seq_len(filtered_past_key_values)
                                    if use_prefix_filter
                                    else current_past_kv
                                )
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                                if kv_log is not None:
                                    log_row = _append_compact_kv_log(
                                        kv_log,
                                        block_idx,
                                        small_block_idx,
                                        block_log_step,
                                        "build_cache",
                                        current_past_kv,
                                        dense_local_kv,
                                        current_past_kv + dense_local_kv,
                                        local_mode=("build_cache_prefix_filter" if use_prefix_filter else "build_cache_no_filter"),
                                        dense_prefix_kv=current_past_kv,
                                        filtered_prefix_kv=filtered_prefix_kv,
                                        dense_local_kv=dense_local_kv,
                                        commit_total_kv=current_past_kv + dense_local_kv,
                                        past_kv_value=(filtered_prefix_kv if use_prefix_filter else current_past_kv),
                                        cache_state=cache_state,
                                        decode_step=decode_step,
                                    )
                            else:
                                # Reuse 阶段是 Sparse-dLLM 机制真正开始省上下文的地方。
                                # 此时 attention 看到的上下文不再是“完整历史 prefix + local block”，
                                # 而是：
                                #   filtered historical prefix KV
                                #   + dense current local block KV
                                #
                                # 这样做保住了局部建块精度，同时削减了远距离 prefix 上下文。
                                if use_prefix_filter:
                                    # cache_position 要继续按“原始 dense 时间轴”来编码当前位置。
                                    # 即使 prefix 被裁剪了，当前 small block 里的 token 仍然应该
                                    # 使用它们在完整序列中的真实位置。
                                    dense_prefix_len = current_past_kv
                                    cache_position = torch.arange(
                                        dense_prefix_len + small_block_start_idx,
                                        dense_prefix_len + small_block_end_idx,
                                        device=self.device,
                                    )
                                    reuse_past_key_values = filtered_past_key_values
                                else:
                                    cache_position = None
                                    reuse_past_key_values = past_key_values
                                output = self.forward(
                                    input_ids=x_t[:, start:end],
                                    use_cache=True,
                                    past_key_values=reuse_past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx,
                                    cache_position=cache_position,
                                )
                                logits = output.logits
                                dense_local_kv = _cache_seq_len(block_past_key_values)
                                filtered_prefix_kv = _cache_seq_len(reuse_past_key_values)
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                if kv_log is not None:
                                    log_row = _append_compact_kv_log(
                                        kv_log,
                                        block_idx,
                                        small_block_idx,
                                        block_log_step,
                                        "reuse_cache",
                                        current_past_kv,
                                        dense_local_kv,
                                        filtered_prefix_kv + dense_local_kv,
                                        local_mode=("reuse_cache_prefix_filter" if use_prefix_filter else "reuse_cache_no_filter"),
                                        dense_prefix_kv=current_past_kv,
                                        filtered_prefix_kv=filtered_prefix_kv,
                                        dense_local_kv=dense_local_kv,
                                        commit_total_kv=current_past_kv + dense_local_kv,
                                        past_kv_value=filtered_prefix_kv,
                                        cache_state=cache_state,
                                        decode_step=decode_step,
                                    )
                            block_log_step += 1
                        else:
                            # 纯 baseline 路径：不做任何 Sparse-dLLM 筛选，也不走 block reuse。
                            output = self.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                            if kv_log is not None:
                                log_row = _append_compact_kv_log(
                                    kv_log,
                                    block_idx,
                                    small_block_idx,
                                    block_log_step,
                                    "dense_no_cache",
                                    current_past_kv,
                                    block_size,
                                    current_past_kv + block_size,
                                    local_mode="baseline_full",
                                    dense_prefix_kv=current_past_kv,
                                    filtered_prefix_kv=current_past_kv,
                                    dense_local_kv=block_size,
                                    commit_total_kv=current_past_kv + block_size,
                                    cache_state=cache_state,
                                    decode_step=decode_step,
                                )
                            block_log_step += 1

                        # Fast-dLLM 原本的并行 reveal 规则保留不变：
                        # 先采样 / 取样候选 token，再依据 threshold 决定这一轮揭示多少 mask。
                        # Sparse-dLLM 改的是“模型看哪些 KV”，而不是“如何决定 unmask 哪些 token”。
                        x_1, p_1t = self.sample_with_top_p(
                            logits,
                            top_p=top_p,
                            temperature=temperature,
                        )
                        x1_p = torch.squeeze(
                            torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)),
                            -1,
                        )
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        # threshold 之上的位置会被并行揭示；
                        # 同时强制保留一个最大概率位置，避免这一轮一个 token 都不揭示。
                        unmask_idx = x1_p > threshold
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]
                        if log_row is not None:
                            log_row["unmask_count"] = int(unmask_idx.sum().item())

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        if stop_token is not None and stop_token >= 0:
                            finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                            finished_flag = finished_flag | finished_row_flags

                        decode_step += 1

            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        mask = seq_block_idx == block_idx
                        input_ids[mask, (block_idx + 1) * block_size] = x_t[
                            mask, (block_idx + 1) * block_size
                        ]

            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)

                keep_mask = ~finished_flag
                sample_indices = sample_indices[keep_mask]
                input_ids = input_ids[keep_mask]
                seq_block_idx = seq_block_idx[keep_mask]
                seq_len = seq_len[keep_mask]
                x_t = x_t[keep_mask]

                if past_key_values is not None:
                    for layer_id in range(len(past_key_values)):
                        past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][keep_mask]
                        past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][keep_mask]

                finished_flag = finished_flag[keep_mask]

        if len(finished_samples) < batch_size and sample_indices.numel() > 0:
            for sample_idx in range(sample_indices.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = input_ids[sample_idx : sample_idx + 1].clone().squeeze(dim=0)

        assert len(finished_samples) == batch_size
        return finished_samples

    @torch.no_grad()
    def mdm_sample_with_visualization(
        self,
        input_ids,
        tokenizer,
        block_size=32,
        max_new_tokens=1024, 
        mask_id=FAST_DLLM_MASK_ID,
        threshold=0.95,
        small_block_size=32,
        stop_token=FAST_DLLM_STOP_TOKEN,
        temperature=0.0,
        top_p=0.95,
    ):
        """
        MDM sampling function with visualization
        with intermediate state output for Gradio visualization
        """
        nfe = 0
        self.model.bd_size = block_size
        num_blocks = max_new_tokens // block_size

        # Initialize state - show all positions as mask
        initial_state = []

        if input_ids.shape[1] > block_size:
            output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True)
            logits, past_key_values = output.logits, output.past_key_values
            nfe += 1
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size
        original_input_length = input_ids.shape[1]

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]

            # Use the length of the first block to initialize state
            first_block_length = block_size - (input_ids.shape[1] % block_size)

            if len(initial_state) == 0:
                for i in range(first_block_length):
                    initial_state.append(("[MASK]", MASK_COLOR))
                yield initial_state
            else:
                for i in range(first_block_length):
                    current_state.append(("[MASK]", MASK_COLOR))
                yield current_state


            # Initialize x_init as mask_id
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)
                
            x_t = x_init.clone()
            block_past_key_values = None
            step = 0
            
            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate next token
                if mask_idx.sum() == 0:
                    nfe += 1
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    token_text = tokenizer.decode([next_token[0].item()], skip_special_tokens=True)
                    # Handle special characters
                    token_text = token_text
                    current_state.append((token_text, TOKEN_COLOR))
                    yield current_state
                    break
                    
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                break

                        logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]
                            
                        step += 1
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)

                        # Select tokens with probability greater than threshold in p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, small_block_start_idx:small_block_end_idx], x1_p, -torch.inf)
                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        # Generate visualization state
                        current_state = []
                        generated_tokens = x_t[0, original_input_length:]
                        
                        # Display generated tokens
                        for i, token_id in enumerate(generated_tokens):
                            if token_id == mask_id:
                                current_state.append(("[MASK]", MASK_COLOR))
                            else:
                                token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
                                # Handle special characters
                                token_text = token_text
                                current_state.append((token_text, TOKEN_COLOR))
                        
                        yield current_state

            input_ids = x_t
            
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
            
        # Final state - display complete text
        final_state = []
        generated_tokens = input_ids[0, original_input_length:]
        for token_id in generated_tokens:
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
            token_text = token_text
            final_state.append((token_text, TOKEN_COLOR))
        
        # Final state doesn't need mask padding, only show actually generated tokens
        
        yield final_state
        
        # Return final text
        final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield final_text


def setup_model_with_custom_generation(model):
    """
    Set up custom generation functions for the model
    """
    # Add mdm_sample method with visualization
    model.mdm_sample_with_visualization = types.MethodType(Fast_dLLM_QwenForCausalLM.mdm_sample_with_visualization, model)
    return model
