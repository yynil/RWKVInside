# RWKVInside

# HybridModel Overview

```mermaid
flowchart TB
    subgraph HybridModel
        input[Input] --> Embeddings
        Embeddings --> L0[Layer 0]
        
        subgraph Layer0[Layer 0]
            direction LR
            A0[AttentionWrapper] --> MLP0[MLP]
            subgraph AW0[AttentionWrapper 0]
                direction TB
                TA0[TeacherAttn\nFrozen] --> |Compare|SA0[StudentAttn\nRWKV TimeMixer]
                SA0 --> |Generate|VF0[v_first state 0]
            end
        end
        
        subgraph Layer1[Layer 1]
            direction LR
            A1[AttentionWrapper] --> MLP1[MLP]
            subgraph AW1[AttentionWrapper 1]
                direction TB
                TA1[TeacherAttn\nFrozen] --> |Compare|SA1[StudentAttn\nRWKV TimeMixer]
                SA1 --> |Generate|VF1[v_first state 1]
            end
        end
        
        subgraph LayerN[Layer N]
            direction LR
            AN[AttentionWrapper] --> MLPN[MLP]
            subgraph AWN[AttentionWrapper N]
                direction TB
                TAN[TeacherAttn\nFrozen] --> |Compare|SAN[StudentAttn\nRWKV TimeMixer]
                SAN --> |Generate|VFN[v_first state N]
            end
        end
        
        L0 --> L1[Layer 1]
        L1 --> LN[...]
        LN --> LayerN
        LayerN --> output[Output]
        
        subgraph VFirstHolder[VFirstHolder Distributed State]
            direction TB
            VS[Shared State\nShape: world_size x batch_size x seq_length x hidden_size]
        end
        
        %% v_first flow connections
        VF0 --> |Update Rank Slice|VS
        VS --> |Next Layer Input|VF1
        VF1 --> |Update Rank Slice|VS
        VS --> |Next Layer Input|VFN
        VFN --> |Update Rank Slice|VS
    end
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef attention fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef vfirst fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef mlp fill:#f1f8e9,stroke:#33691e,stroke-width:2px;
    
    class A0,A1,AN attention;
    class VF0,VF1,VFN,VS vfirst;
    class MLP0,MLP1,MLPN mlp;
```

## Training Process

```mermaid
graph TD
    A[开始] --> B[参数初始化]
    B --> C[加载配置文件]
    C --> D[初始化模型和分词器]
    D --> E[设置DeepSpeed配置]
    
    E --> F{判断训练阶段<br>stage 1/2/3}
    
    F -->|Stage 1| G1[初始化教师注意力列表]
    F -->|Stage 2| G2[初始化完整教师模型]
    F -->|Stage 3/SFT| G3[跳过教师模型初始化]
    
    G1 --> H[准备数据加载器]
    G2 --> H
    G3 --> H
    
    H --> I[开始训练循环]
    
    subgraph 训练循环
        I --> J[更新学习率和权重衰减]
        J --> K[前向传播]
        K --> L[计算损失]
        L --> M[反向传播]
        M --> N{是否累积步骤}
        N -->|是| O[优化器步进]
        N -->|否| P[继续下一批次]
        O --> Q{检查保存条件}
        P --> J
        Q -->|满足| R[保存检查点]
        Q -->|不满足| S{检查终止条件}
        R --> S
        S -->|继续| J
        S -->|终止| T[结束训练]
    end
    
    subgraph 损失计算
        L --> L1[教师损失]
        L --> L2[KL散度损失]
        L --> L3[学生交叉熵损失]
    end
```