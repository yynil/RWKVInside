import sys
import os
import threading
import gc

def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)
    # rwkv_path = os.path.join(parent_dir, 'rwkv7')
    # sys.path.append(rwkv_path)
    rwkv_llama_path = os.path.join(parent_dir, "rwkv_llama")
    sys.path.append(rwkv_llama_path)
    # print(f'add path: {rwkv_path} to sys.path')
    print(f"add path: {rwkv_llama_path} to sys.path")
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_T_MAX"] = "4096"
    os.environ["RWKV_FLOAT_MODE"] = "bf16"
    os.environ["RWKV_HEAD_SIZE_A"] = "64"
    os.environ["RWKV_T_MAX"] = "4096"
    os.environ["RWKV_MY_TESTING"] = "x060"
    os.environ["RWKV_CTXLEN"] = "4096"
    os.environ["WKV"] = "fla"
    os.environ["RWKV_TRAIN_TYPE"] = ""


setup_env()
from einops import rearrange

# from fla.ops.rwkv6 import chunk_rwkv6,fused_recurrent_rwkv6
import math
from rwkvfla.ops.rwkv7 import fused_recurrent_rwkv7


def RUN_CUDA_RWKV7_STATE(B, T, C, H, r, k, v, w, a, b, s):
    getattr(torch, r.device.type).set_device(r.device.index)
    r = rearrange(r, "b l (h d) -> b h l d", h=H)
    k = rearrange(k, "b l (h d) -> b h l d", h=H)
    v = rearrange(v, "b l (h d) -> b h l d", h=H)
    w = rearrange(w, "b l (h d) -> b h l d", h=H)
    a = rearrange(a, "b l (h d) -> b h l d", h=H)
    b = rearrange(b, "b l (h d) -> b h l d", h=H)

    o, state = fused_recurrent_rwkv7(
        r,
        k,
        v,
        w,
        a,
        b,
        scale=1.0,
        initial_state=s,
        output_final_state=True,
        training=False,
    )
    x = rearrange(o, "b h l d -> b l (h d)")
    return x, state


import torch
from utilities import TimeMixState, ChannelMixState, BlockState
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

# from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import os

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id, update_v_first, get_v_first):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_embd = args.n_embd

        self.update_v_first = update_v_first
        self.get_v_first = get_v_first

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
            )
            self.x_v = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)
            )
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = 64
            # D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (
                    0.85 + 1.0 * ratio_0_to_1**0.5
                )
            self.w0 = nn.Parameter(
                decay_speed.reshape(1, 1, C) + 0.5
            )  # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 64
            # D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            D_MV_LORA = 32
            # D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            D_GATE_LORA = 128
            # D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            if not self.args.gate_free:
                self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
                self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            if self.args.has_group_norm:
                self.ln_x = nn.GroupNorm(
                    H, C, eps=(1e-5) * (args.head_size_divisor**2)
                )  # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()

    def forward(self, x, last_state: TimeMixState):
        shift_state = last_state.shift_state
        B, T, C = x.size()
        H = self.n_head
        if shift_state is not None:
            xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        else:
            xx = self.time_shift(x) - x
        lx = x[:, -1]

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            self.update_v_first(v)
        else:
            ### Original implementation
            v = v + (self.get_v_first().to(v.device) - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )  # add value residual

        a = torch.sigmoid(
            self.a0 + (xa @ self.a1) @ self.a2
        )  # a is "in-context learning rate"

        if not self.args.gate_free:
            g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        wkv_state = last_state.wkv_state
        x, wkv_state = RUN_CUDA_RWKV7_STATE(
            B,
            T,
            C,
            H,
            r.bfloat16(),
            k.bfloat16(),
            v.bfloat16(),
            w.bfloat16(),
            -kk.bfloat16(),
            (kk * a).bfloat16(),
            s=wkv_state,
        )
        if self.args.has_group_norm:
            x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        if not self.args.gate_free:
            x = self.output(x * g)
        else:
            x = self.output(x)
        return x, TimeMixState(lx, wkv_state)


class RWKV_Tmix_x070_Wrapper(nn.Module):
    def __init__(self, args, layer_id, update_v_first, get_v_first):
        super().__init__()
        self.args = args
        self.layer_idx = layer_id
        self.time_mixer = RWKV_Tmix_x070(args, layer_id, update_v_first, get_v_first)

    def forward(self, hidden_states, past_key_value, **kwargs):
        x = hidden_states
        args = self.args
        B, T, C = x.size()
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                last_state = None
            else:
                last_state = past_key_value[self.layer_idx][0]
        if last_state is None:
            H = args.dim_att // args.head_size_a
            device = x.device
            dtype = x.dtype
            wkv_states = torch.empty((B, H, C // H, C // H), device=device, dtype=dtype)
            token_shift = torch.empty((B, C), device=device, dtype=dtype)
            wkv_states[:] = 0
            token_shift[:] = 0
            time_state = TimeMixState(token_shift, wkv_states)
            # print(wkv_states)
            channel_state = None
            last_state = BlockState(time_state, channel_state)
        x, states = self.time_mixer(x, last_state.time_mix_state)
        last_state.time_mix_state = states
        if past_key_value is not None:
            keys = T
            values = last_state
            past_key_value.update(keys, values, self.layer_idx)
        return x, None


class HybridModel(nn.Module):
    def __init__(self, rwkv_args, transformer_config):
        super(HybridModel, self).__init__()
        self.args = rwkv_args
        print(f"rwkv_args: {rwkv_args}")
        print(f"transformer_config: {transformer_config}")
        if transformer_config.tie_word_embeddings:
            transformer_config.tie_word_embeddings = False
        with no_init_weights():
            self.model = AutoModelForCausalLM.from_config(transformer_config)
        print(f"init transformer model: {self.model}")

        # Register v_first as a buffer
        self.thread_local = threading.local()
        self.thread_local.v_first = None

        # Replace the self attention to TimeMixer
        for layer_idx in range(transformer_config.num_hidden_layers):
            llama_layer = self.model.model.layers[layer_idx]
            if layer_idx in rwkv_args.layers:
                att = RWKV_Tmix_x070_Wrapper(
                    rwkv_args,
                    layer_idx,
                    self.update_v_first,  # Pass the callback function
                    self.get_v_first,
                )
                old_attn = llama_layer.self_attn
                llama_layer.self_attn = att
                del old_attn
                print(f"layer {layer_idx} is replaced by RWKV TimeMixer_x070")

        import gc

        gc.collect()

    def forward(
        self,
        input_ids,
        inference_params=None,
        **kwargs,
    ):
        # Initialize v_first as None for the first layer
        kwargs["v_first"] = None
        return self.model(input_ids, **kwargs)

    def update_v_first(self, new_v_first):
        """Callback function to update v_first in HybridModel."""
        self.thread_local.v_first = new_v_first

    def get_v_first(self):
        return self.thread_local.v_first

    def load_checkpoint(self, path):
        all_keys = set(self.state_dict().keys())
        incompatible_keys = set()
        #if the path is the file, load it directly
        #if the path is the directory, load the sharded files in the directory with suffix .pt
        if os.path.isdir(path):
            files = os.listdir(path)
            files = [os.path.join(path, f) for f in files if f.endswith('.pt')]
        else:
            files = [path]
        for file in files:
            checkpoint = torch.load(file, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)
            print(f'load model from {file}')
            ckpt_keys = checkpoint.keys()
            #subtract the keys in the checkpoint from the all_keys
            #if the ckpt_key exists in the all_keys, remove it
            for ckpt_key in ckpt_keys:
                if ckpt_key in all_keys:
                    all_keys.remove(ckpt_key)
                else:
                    incompatible_keys.add(ckpt_key)
            del checkpoint
            gc.collect()
        print(f'Finish loading model from {path}')
        print(f'Incompatible keys: {incompatible_keys} missing keys: {all_keys}')
        


def create_rwkv_args(transformer_config, config):
    from argparse import Namespace

    args = Namespace()
    args.layers = config["RWKV"]["layers"]
    args.my_pos_emb = 0
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = transformer_config.num_hidden_layers
    args.n_embd = transformer_config.hidden_size
    args.dim_att = transformer_config.hidden_size
    args.dim_ffn = transformer_config.intermediate_size
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_config.vocab_size
    args.pad_id = transformer_config.pad_token_id
    args.is_llama_ffn = config.get("is_llama_ffn", False)
    args.is_rwkv_att_only = config.get("is_rwkv_att_only", False)
    return args


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config_file = "configs/qwen_7b.yaml"
    import yaml

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    from transformers import AutoConfig

    model_id = config["Llama"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_config = AutoConfig.from_pretrained(model_id)
    print(transformer_config)
    args = create_rwkv_args(transformer_config, config)
    model = HybridModel(args, transformer_config)
    print(model)
    ckpt_file = "/home/yueyulin/model/qwen_7b_distill/7b_stage2_model_converted.bin"
    model.load_checkpoint(ckpt_file)
