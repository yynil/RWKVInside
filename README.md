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
### Training shell

The training shell is the train.sh. Please  refer the train_memo.txt for more details for training in different stages.

## RL Training Process

```mermaid
flowchart TB
    subgraph Init["初始化阶段"]
        A[解析命令行参数] --> B[设置分布式环境]
        B --> C[加载分词器]
        C --> D[加载数据集]
        D --> E[创建DataLoader]
        E --> F[初始化DeepSpeed配置]
    end

    subgraph Models["模型初始化"]
        F --> G[初始化主模型]
        G --> H[初始化参考模型]
        H --> J[配置优化器]
    end

    subgraph Training["训练循环"]
        J --> K[进入训练轮次]
        K --> L[批次训练]
        L --> M{是否需要保存检查点}
        M -->|是| N[保存模型检查点]
        M -->|否| O[继续训练]
        N --> O
        O --> P{轮次结束?}
        P -->|否| L
        P -->|是| Q[保存轮次检查点]
        Q --> R{全部轮次完成?}
        R -->|否| K
        R -->|是| S[训练结束]
    end

    subgraph TrainStep["批次训练步骤(GRPOTrainer)"]
        L --> T[生成完成内容]
        T --> U[计算奖励]
        U --> V[计算KL散度]
        V --> W[多次更新迭代]
        W --> X[返回训练指标]
    end

    subgraph Metrics["指标记录"]
        X --> Y[更新累计统计]
        Y --> Z[记录到WandB]
    end
```

### GRPO Algorithm from https://arxiv.org/pdf/2402.03300v3

![alt text](image-1.png)

The implementation of GRPO algorithm is rl/grpo_trainer.py , please refer the code for more details. TRL just eliminates the GRPO iteration and ignore the memory efficiency for large data in relative small machines. That's why we implement our own GRPO algorithm.