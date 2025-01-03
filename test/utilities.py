from typing import Any, Dict, Optional, Union
from transformers.cache_utils import Cache, DynamicCache
import torch
######state
class TimeMixState:
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:
    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, H, device, dtype):
        result = BlockStateList.empty(N, B, C, H, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        wkv_states = torch.empty((N, B, H, C//H, C//H),
                                 device=device,
                                 dtype=torch.bfloat16)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state

class HybridCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        self.rwkv_layers = set()

    def __repr__(self) -> str:
        rwkv_layers =  f"HybridCache(rwkv_layers={self.rwkv_layers})"
        #count the number of key_cache and value_cache
        key_cache_count = sum(len(cache) for cache in self.key_cache)
        value_cache_count = sum(len(cache) for cache in self.value_cache)
        count_info = rwkv_layers + f", key_cache_count={key_cache_count}, value_cache_count={value_cache_count}"
        memories = 0
        seq_length = self.get_seq_length()
        for cache in self.value_cache:
            for data in cache:
                if not isinstance(data,torch.Tensor):
                    memories += data.time_mix_state.wkv_state.numel()
                else:
                    memories += data.numel()
        count_info += f", memories={memories/1024/1024}MB, seq_length={seq_length}"
        return count_info
    def update(self, 
               key_states: Union[int,torch.Tensor], 
               value_states: Union[torch.Tensor,BlockState], 
               layer_idx :int, 
               cache_kwargs: Optional[Dict[str, Any]] = None):
        if isinstance(key_states,int) and not isinstance(value_states, torch.Tensor):
            self.rwkv_layers.add(layer_idx)
            # 如果该 layer_idx 的缓存为空，则创建一个新的列表
            if layer_idx >= len(self.key_cache):
                self.key_cache.append([])
                self.value_cache.append([])
            
            # 如果该 layer_idx 的第 0 个状态是空，则插入一个新的状态
            if len(self.key_cache[layer_idx]) == 0:
                self.key_cache[layer_idx].append(key_states)
                self.value_cache[layer_idx].append(value_states)
            else:
                # 如果不是空，则覆盖第 0 个状态
                self.key_cache[layer_idx][0] = self.key_cache[layer_idx][0]+key_states
                self.value_cache[layer_idx][0] = value_states
            
            return key_states,value_states
        
        # 对于其他情况，使用原有的更新逻辑
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
    
    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # print("来自 HybridCache: get_seq_length 方法被调用")
        if layer_idx in self.rwkv_layers:
            return self.key_cache[layer_idx][0]
        return super().get_seq_length(layer_idx)
    
    def get_max_length(self):
        # print("来自 HybridCache: get_max_length 方法被调用")
        return super().get_max_length()
    
    def reorder_cache(self, beam_idx):
        # print("来自 HybridCache: reorder_cache 方法被调用")
        return super().reorder_cache(beam_idx)
    
    def __getitem__(self, item):
        # print("来自 HybridCache: __getitem__ 方法被调用")
        if item in self.rwkv_layers:
            return self.value_cache[item]
        return super().__getitem__(item)
    def offload_to_cpu(self):
        for cache in self.value_cache:
            for data in cache:
                if isinstance(data,torch.Tensor):
                    data.cpu()
                else:
                    data.time_mix_state.wkv_state.cpu()
                    data.time_mix_state.shift_state.cpu()
                    
    def offload_to_cuda(self,device: str):
        for cache in self.value_cache:
            for data in cache:
                if isinstance(data,torch.Tensor):
                    data.cuda(device)
                else:
                    data.time_mix_state.wkv_state.cuda(device)
                    data.time_mix_state.shift_state.cuda(device)