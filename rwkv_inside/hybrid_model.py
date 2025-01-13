import sys
import os
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)
    # rwkv_path = os.path.join(parent_dir, 'rwkv7')
    # sys.path.append(rwkv_path)
    rwkv6_path = os.path.join(parent_dir, 'rwkv')
    sys.path.append(rwkv6_path)
    rwkv_llama_path = os.path.join(parent_dir, 'rwkv_llama')
    sys.path.append(rwkv_llama_path)
    # print(f'add path: {rwkv_path} to sys.path')
    print(f'add path: {parent_dir} to sys.path')
    print(f'add path: {rwkv_llama_path} to sys.path')
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = os.environ.get('RWKV_T_MAX', '4096')
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    
    os.environ['RWKV_CTXLEN'] = os.environ.get('RWKV_CTXLEN', '4096')
    if 'WKV' not in os.environ:
        os.environ['WKV'] = ''
    if "RWKV_TRAIN_TYPE" not in os.environ:
        os.environ["RWKV_TRAIN_TYPE"] = ''
    RWKV_VERSION = os.environ.get('RWKV_VERSION', 'v7')
    if RWKV_VERSION == 'v7':
        os.environ["RWKV_MY_TESTING"]='x070'
    else:
        os.environ["RWKV_MY_TESTING"]='x060'
    print(f'RWKV_VERSION is {RWKV_VERSION}')
    
setup_env()

from typing import Optional, Tuple
RWKV_VERSION=os.environ.get('RWKV_VERSION','v7')
is_rwkv_7 = RWKV_VERSION == 'v7'
if is_rwkv_7 :
    from TimeMixer import RWKV_Tmix_x070 as TimeMixer
else:
    from TimeMixer import RWKV_Tmix_x060 as TimeMixer
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM
import gc
import logging
import deepspeed
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class VFirstHolder(nn.Module):
    
    def __init__(self, batch_size: int, seq_length: int, hidden_size: int,world_size: int,dtype=torch.bfloat16):
        super().__init__()
        self.shared_state = nn.Parameter(
            torch.zeros(
                (world_size,batch_size, seq_length, hidden_size),
                dtype=dtype
            ),
            requires_grad=False
        )
    


class AttentionWrapper(nn.Module):
    
    def __init__(self,student_attn,layer_idx,args):
        super(AttentionWrapper, self).__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.teacher_attn = None
        self.student_attn = student_attn
        self.student_attn.requires_grad_(True)
        self.add_module("student_attn", self.student_attn)
        self.v_first_state = None#v6 will benefit from v_first_state
        self.global_rank = None
    
    def forward(self, 
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        kwargs['output_attentions'] = False

        # NOTE - instead of returning attentions here we return a special attention loss
        hidden_states = kwargs['hidden_states']
        past_key_value = kwargs.get("past_key_value", None)
        hidden_states = hidden_states.requires_grad_(True)
        v_first = self.v_first_state.shared_state.data[self.global_rank].clone()
        if self.args.grad_cp == 1:
            if is_rwkv_7:
                student_hidden_states,v_first = deepspeed.checkpointing.checkpoint(self.student_attn, hidden_states, v_first)
            else:
                student_hidden_states = deepspeed.checkpointing.checkpoint(self.student_attn, hidden_states)
        else:
            if is_rwkv_7:
                student_hidden_states,v_first = self.student_attn(hidden_states, v_first)
            else:
                student_hidden_states = self.student_attn(hidden_states)
        self.v_first_state.shared_state.data[self.global_rank].copy_(v_first)
        if self.args.stage != 1:
            return (student_hidden_states, None, past_key_value)
        # student_outputs = self.student_attn(hidden_states)
        with torch.no_grad():
            teacher_outputs = self.teacher_attn(*args, **kwargs)
        # special attention loss is the vector norm of the difference between the student and teacher attn outputs
        # student_hidden_states = student_outputs[0]
        teacher_hidden_states = teacher_outputs[0]
        special_attn_loss = torch.linalg.vector_norm(teacher_hidden_states - student_hidden_states, dim=-1).mean() * (teacher_hidden_states[0].size(-1) ** -0.5)
        return (teacher_outputs[0], special_attn_loss, ) + teacher_outputs[2:]

class HybridModel(nn.Module):
    
    @staticmethod
    def get_rwkv_args(transformer_config):
        from argparse import Namespace
        args = Namespace()
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
        args.layers = [i for i in range(transformer_config.num_hidden_layers)]
        args.pad_id = transformer_config.eos_token_id
        args.stage = 4
        args.is_rwkv_att_only = True
        args.is_all_labels_kl = True
        args.init_with_llama = False
        return args
    
    def __init__(self, transformer_model, rwkv_args, tokenizer=None):
        super(HybridModel, self).__init__()
        stage = rwkv_args.stage
        if stage == 1:
            #Freeze the model
            transformer_model.requires_grad_(False)
        else:
            #Unfreeze the model
            transformer_model.requires_grad_(True)
            if transformer_model.config.tie_word_embeddings:
                # copy untied embeddings
                transformer_model.get_output_embeddings().weight = nn.Parameter(transformer_model.get_input_embeddings().weight.clone())
                # untie the embeddings in the config, too
                transformer_model.tie_word_embeddings = False

        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                #Only replace the attention layer with TimeMixer
                student_attn = TimeMixer(rwkv_args, layer_idx)
                llama_layer = transformer_model.model.layers[layer_idx]
                attn_wrapper = AttentionWrapper(student_attn,layer_idx,rwkv_args)
                llama_layer.self_attn = attn_wrapper
                gc.collect()
        self.model = transformer_model
        self.add_module("model", self.model)
        self.args = rwkv_args
        self.teacher_model = None  # 初始化为None，后续再设置
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            if 'pad_token_id' not in self.tokenizer.__dict__:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        torch.cuda.empty_cache()
        self.client = None

 
    def forward(
        self,
        input_ids,
        **kwargs,
    ):
        ret = self.model(input_ids, **kwargs)
        return ret
    
    def load_check_point(self, path):
        all_keys = set(self.state_dict().keys())
        incompatible_keys = set()
        #if the path is the file, load it directly
        #if the path is the directory, load the sharded files in the directory with suffix .pt or .bin
        if os.path.isdir(path):
            files = os.listdir(path)
            files = [os.path.join(path, f) for f in files if f.endswith('.pt') or f.endswith('.bin')]
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
        
        
        return


if __name__ == '__main__':
    model_id = '/home/yueyulin/models/Qwen2.5-0.5B-Instruct/'
    from transformers.modeling_utils import no_init_weights
    config = AutoConfig.from_pretrained(model_id)
    rwkv_args = HybridModel.get_rwkv_args(config)
    rwkv_args.has_group_norm = True
    with no_init_weights():
        transformer_model = AutoModelForCausalLM.from_config(config)
    hybrid_model = HybridModel(transformer_model, rwkv_args)    
    print(hybrid_model)
    ckpt_path = '/home/yueyulin/model/qwen_0.5b_full_layers_stage2_v7_finemath/pytorch_model.bin'
    hybrid_model.load_check_point(ckpt_path)

