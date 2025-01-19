# test_mask.py
import torch
from TimeMixer import RUN_CUDA_RWKV7g

def test_wkv_with_mask():
    print("Running test_wkv_with_mask...")
    # 创建测试数据
    B, T, HC = 2, 32, 128  # batch_size=2, seq_len=32, hidden_size=128
    device = "cuda:0"
    # 创建输入tensor
    tensors = {
        name: torch.randn(B, T, HC, dtype=torch.bfloat16, device=device).contiguous()
        for name in ['q', 'w', 'k', 'v', 'a', 'b']
    }
    
    # 创建mask
    mask = torch.ones(B, T, dtype=torch.int32, device=device)
    # mask第二个序列的后半部分
    mask[1, T//2:] = 0
    
    print("Running without mask...")
    output1 = RUN_CUDA_RWKV7g(
        tensors['q'], tensors['w'], tensors['k'],
        tensors['v'], tensors['a'], tensors['b']
    )
    
    print("Running with mask...")
    output2 = RUN_CUDA_RWKV7g(
        tensors['q'], tensors['w'], tensors['k'],
        tensors['v'], tensors['a'], tensors['b'],
        mask
    )
    
    # 验证结果
    print("\nValidating results:")
    # 检查第一个序列是否相同(未mask)
    first_seq_same = torch.allclose(output1[0], output2[0], rtol=1e-3)
    print(f"First sequence unchanged: {first_seq_same}")
    
    # 检查第二个序列的masked部分是否不同
    masked_changed = not torch.allclose(
        output1[1, T//2:], 
        output2[1, T//2:], 
        rtol=1e-3
    )
    print(f"Masked part changed: {masked_changed}")
    
    return first_seq_same and masked_changed

if __name__ == "__main__":
    success = test_wkv_with_mask()
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")