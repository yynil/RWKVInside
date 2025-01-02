#include <cuda_bf16.h>
#include <assert.h>

using bf = __nv_bfloat16;
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return 	__float2bfloat16_rn(u); }

typedef bf * __restrict__ F_;

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, bf* y_, float* s_, float* sa_) {
    constexpr int C = _C_;
    int bind = blockIdx.y, hind = blockIdx.x, i = threadIdx.x;

    float state[C] = {};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    for (int t = 0; t < T; t++) {
        int ind = bind*T*H*C + t*H*C + hind * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;

        float v = to_float(v_[ind]);
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + k[j] * v + sa * b[j];
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bind*H+hind)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }
    }
}

__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s_, float * __restrict__ sa_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    constexpr int C = _C_;
    int bind = blockIdx.y, hind = blockIdx.x, i = threadIdx.x;

    float stateT[C] = {};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];

    extern __shared__ char smem_[];
    float*dstate = (float*)smem_; //[C*(C+1)];

    for (int j = 0; j < C; j++) {
        dstate[i*(C+1)+j] = 0;
    }

    for (int t = T-1; t >= 0; t--) {
        int ind = bind*T*H*C + t*H*C + hind * C + i;
        float bi, ki, dyi, wi;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bind*H+hind)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = to_bf(dq);

        float iwi = 1.f/wi;
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - bi*sa[j] - ki*v[j]) * iwi;
            dstate[i*(C+1)+j] += dyi * q[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float ds = dstate[j*(C+1)+i];
            dw += ds*stateT[j];
            dk += ds*v[j];
            db += ds*sa[j];
        }
#pragma unroll
        for (int j = 0; j < C; j++) {
            float ds = dstate[i*(C+1)+j];
            dv += ds*k[j];
            dSb += ds*b[j];
        }
        dw_[ind] = to_bf(dw * wi * wi_fac);
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[ind] = to_bf(da);

        for (int j = 0; j < C; j++) {
            dstate[i*(C+1)+j] = dstate[i*(C+1)+j]*w[j] + dSb * a[j];
        }
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa) {
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa);
}
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) {
    assert(T%_CHUNK_LEN_ == 0);
    int shared_mem = _C_*(_C_+1)*4;
    assert(!cudaFuncSetAttribute(backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
    backward_kernel<<<dim3(H,B), dim3(_C_), shared_mem>>>(T,H,w,q,k,v,z,a,dy,s,sa,dw,dq,dk,dv,dz,da);
}
