from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

# 解释为什么没有cfg_interval_steps
@dataclass
class DllmCacheConfig:
    prompt_interval_steps: int = 1
    gen_interval_steps: int = 1
    transfer_ratio: float = 0.0

# 搬运cache_hook_Dream.py和cache_hook_LLaDA.py中的refresh_index函数
def refresh_index(
    new_features: torch.Tensor,
    cached_features: torch.Tensor = None,
    transfer_ratio: float = 0.5,
    layer_id: int = 0,
) -> torch.Tensor:
    batch_size, gen_len, d_model = new_features.shape
    num_replace = int(gen_len * transfer_ratio)
    cos_sim = torch.nn.functional.cosine_similarity(new_features, cached_features, dim=-1)
    transfer_index = torch.topk(cos_sim, largest=False, k=num_replace).indices
    return transfer_index


class DllmCacheRuntime:
    """V2-equivalent dLLM-Cache runtime container for C_p / C_r."""

    def __init__(self, config: DllmCacheConfig):
        self.config = config
        self.prompt_length = 0
        self._cache: dict[str, dict[int, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        self._step_counter: dict[int, int] = defaultdict(int)

    @classmethod
    def new_instance(
        cls,
        prompt_interval_steps: int = 1,
        gen_interval_steps: int = 1,
        transfer_ratio: float = 0.0,
    ) -> "DllmCacheRuntime":
        return cls(
            DllmCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )

    def reset(self, prompt_length: int = 0) -> None:
        self.prompt_length = int(prompt_length)
        self._cache = defaultdict(lambda: defaultdict(dict))
        self._step_counter = defaultdict(int)

    def start_new_block(self, reset_steps: bool = True) -> None:
        self._cache["gen"] = defaultdict(dict)
        if reset_steps:
            self._step_counter = defaultdict(int)

    # update_step / refresh_prompt / refresh_gen / current_step / set_cache / get_cache 对应原始 Cache 状态机
    # 参考Cache.py中的实现
    def update_step(self, layer_id: int) -> None:
        self._step_counter[layer_id] += 1

    @property
    def current_step(self) -> int:
        return max(self._step_counter.values(), default=1)

    def _interval_hit(self, interval: int) -> bool:
        if interval is None or interval <= 0:
            return False
        return (self.current_step - 1) % int(interval) == 0

    def refresh_gen(self, layer_id: int = 0) -> bool:
        return self._interval_hit(self.config.gen_interval_steps)

    def refresh_prompt(self, layer_id: int = 0) -> bool:
        return self._interval_hit(self.config.prompt_interval_steps)

    @property
    def transfer_ratio(self) -> float:
        return float(self.config.transfer_ratio)

    def set_cache(self, layer_id: int, feature_name: str, features: Any, cache_type: str) -> None:
        self._cache[cache_type][layer_id][feature_name] = features

    def get_cache(self, layer_id: int, feature_name: str, cache_type: str) -> Any | None:
        return self._cache.get(cache_type, {}).get(layer_id, {}).get(feature_name)

    def has_cache(self, layer_id: int, feature_name: str, cache_type: str) -> bool:
        return feature_name in self._cache.get(cache_type, {}).get(layer_id, {})

    def shrink_batch(self, keep_mask: torch.Tensor) -> None:
        for cache_type, by_layer in list(self._cache.items()):
            for layer_id, feature_map in list(by_layer.items()):
                for feature_name, feature_data in list(feature_map.items()):
                    if isinstance(feature_data, dict):
                        for key, tensor in list(feature_data.items()):
                            if isinstance(tensor, torch.Tensor):
                                feature_data[key] = tensor[keep_mask]
                    elif isinstance(feature_data, torch.Tensor):
                        feature_map[feature_name] = feature_data[keep_mask]

    def sync_prompt_kv_from_past(self, layer_id: int, past_key_value: Any) -> None:
        if past_key_value is None or len(past_key_value) <= layer_id:
            return
        k_prompt, v_prompt = past_key_value[layer_id]
        self.set_cache(
            layer_id=layer_id,
            feature_name="kv_cache",
            features={"k": k_prompt, "v": v_prompt},
            cache_type="prompt",
        )
