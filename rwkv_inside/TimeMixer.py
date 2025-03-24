import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
HEAD_SIZE = 64
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
print(f'parent_dir: {parent_dir}')
is_wind_cuda = False
if 'is_wind_cuda' in os.environ:
    is_wind_cuda = os.environ['is_wind_cuda'] == '1'
if is_wind_cuda:    
    load(name="wind", sources=[f'{parent_dir}/rwkv_cuda_wind/wind_rwkv7.cu', f'{parent_dir}/rwkv_cuda_wind/wind_rwkv7.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=[f'-D_C_={HEAD_SIZE}',"-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"])

    class WindRWKV7(torch.autograd.Function):
        @staticmethod
        def forward(ctx,w,q,k,v,a,b):
            B,T,H,C = w.shape
            s0 = torch.zeros(B,H,C,C,dtype=w.dtype,device=w.device)
            assert T%16 == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,a,b,s0])
            w,q,k,v,a,b,s0 = [i.contiguous() for i in [w,q,k,v,a,b,s0]]
            y = torch.empty_like(v)
            sT = torch.empty_like(s0)
            s = torch.zeros(B,H,T//16,C,C, dtype=w.dtype,device=w.device)
            torch.ops.wind.forward(w,q,k,v,a,b, s0,y,s,sT)
            ctx.save_for_backward(w,q,k,v,a,b,s)
            return y
        
        @staticmethod
        def backward(ctx,dy):
            w,q,k,v,a,b,s = ctx.saved_tensors
            B,T,H,C = w.shape
            dsT = torch.zeros(B,H,C,C,dtype=dy.dtype,device=dy.device)
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            dy,dsT = [i.contiguous() for i in [dy,dsT]]
            dw,dq,dk,dv,da,db,ds0 = [torch.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
            torch.ops.wind.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
            return dw,dq,dk,dv,da,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [q,w,k,v,a,b]]
        return WindRWKV7.apply(w,q,k,v,a,b).view(B,T,HC)
else:#use fast
    CHUNK_LEN = 16
    print(f"Loading pre-built extension for RWKV7 with CUDA")
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    module = load(name="wind_backstepping", sources=[f'{parent_dir}/cuda/wkv7_cuda.cu', f'{parent_dir}/cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    print(f"Loaded pre-built extension successfully, module: {module}")
    # try:
    #     # 尝试加载已编译的扩展
    #     torch.ops.load_library("wind_backstepping")
    #     print("Loaded pre-built extension successfully")
    # except Exception as e:
    #     print(f"Error loading extension: {e}")
    #     print("Please run 'python setup.py build_ext --inplace' first")
    #     raise

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b, mask=None):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            # Check mask if provided
            if mask is not None:
                assert mask.shape == (B, T)
                assert mask.dtype == torch.int32
                assert mask.is_contiguous()
        
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa,mask)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa,mask)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa,mask = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db,mask)
            return dw,dq,dk,dv,dz,db,None

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b, mask=None):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b,mask).view(B,T,HC)
    
class RWKV_Tmix_x070(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
   
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        self.gate_free = args.gate_free if hasattr(args, 'gate_free') else False
        self.has_group_norm = args.has_group_norm if hasattr(args, 'has_group_norm') else False

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            # self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(torch.ones(1, 1, C))#make x_g to be 1.0 as initialized values
            

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
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
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 64
            # D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = 32
            # D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            D_GATE_LORA = 128
            # D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            if not self.gate_free:
                self.g1 = nn.Parameter(torch.ones(C, D_GATE_LORA)* 10.0)
                self.g2 = nn.Parameter(torch.ones(D_GATE_LORA, C))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            if self.has_group_norm:
                self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()
            

    
    
    def forward(self, x, v_first,attention_mask):
        # print(f'pid is {os.getpid()} attention_mask: {attention_mask.dtype}')
        B, T, C = x.size()
        H = self.n_head
        # Check if input tensor has NaN
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        # if self.args.local_rank == 3:
        #     v_first_mean = v_first.mean().item()
        #     print(f'pid is {os.getpid()} has_norm:{self.has_group_norm} v_first: {v_first_mean} at layer {self.layer_id}')
        # if self.args.local_rank == 0 and self.layer_id ==1 :
        #     v_first_mean = v_first.mean().item()
        #     print(f'MASTER pid is {os.getpid()} has_norm:{self.has_group_norm} v_first: {v_first_mean} at layer {self.layer_id}')
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        if not self.gate_free:
            g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a,attention_mask)
        if self.has_group_norm:
            x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        else:
            x = x.view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        if not self.gate_free:
            x = self.output(x * g)
        else:
            x = self.output(x)
        return x, v_first
    

####################RWKV 6####################
if "WKV" in os.environ and os.environ["WKV"] == 'fla':
    from einops import rearrange
    from fla.ops.rwkv6 import chunk_rwkv6
    def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
        r = rearrange(r, 'b l (h d) -> b h l d', h = H)
        k = rearrange(k, 'b l (h d) -> b h l d', h = H)
        v = rearrange(v, 'b l (h d) -> b h l d', h = H)
        w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
        o,_ = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=None, output_final_state=False)
        x = rearrange(o, 'b h l d -> b l (h d)')
        return x

else:
    from torch.utils.cpp_extension import load
    HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE_A", 64))
    cuda_dir = os.path.join(parent_dir, 'cuda')
    ctx_len = int(os.environ.get('RWKV_CTXLEN','4096'))
    wkv6_cuda = load(name="wkv6", sources=[f"{cuda_dir}/wkv6_op.cpp", f"{cuda_dir}/wkv6_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={ctx_len}"])
        
    class WKV_6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.bfloat16
                assert k.dtype == torch.bfloat16
                assert v.dtype == torch.bfloat16
                assert w.dtype == torch.bfloat16
                assert u.dtype == torch.bfloat16
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ew = (-torch.exp(w.float())).contiguous()
                ctx.save_for_backward(r, k, v, ew, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.bfloat16
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
        return WKV_6.apply(B, T, C, H, r, k, v, w, u)
    
class RWKV_Tmix_x060(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.has_group_norm = args.has_group_norm if hasattr(args, 'has_group_norm') else False
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
            if args.n_embd==4096:
                TIME_MIX_EXTRA_DIM = TIME_MIX_EXTRA_DIM*2
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, args.n_embd).uniform_(-1e-4, 1e-4))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            TIME_DECAY_EXTRA_DIM = 64
            if args.n_embd==4096:
                TIME_DECAY_EXTRA_DIM = TIME_DECAY_EXTRA_DIM*2
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, args.dim_att).uniform_(-1e-4, 1e-4))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        #init gate parameters to 1
        self.gate.weight.data = self.gate.weight.data * 0 + 1
        if self.has_group_norm:
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        if self.has_group_norm:
            x = self.ln_x(x).view(B, T, C)
        else:
            x = x.view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)
