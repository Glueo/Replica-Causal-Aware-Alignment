# config.py
from dataclasses import dataclass

@dataclass
class CAAConfig:
    # 模型设置
    model_name: str = "gpt2"  # 论文中使用 GPT-2 作为 Base 模型 [cite: 236]
    ref_model_name: str = "gpt2"
    
    # 奖励模型路径 (这里以情感分析为例，对应论文实验1)
    reward_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment" 

    # 训练超参数 [cite: 293-296]
    learning_rate: float = 1e-5        # 论文指定 LR [cite: 593]
    batch_size: int = 64               # 常用设置
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    
    # PPO 参数
    clip_range: float = 0.2            # 论文公式(3)中的 epsilon [cite: 294]
    kl_coef: float = 0.3               # 论文公式(2)中的 beta (文本续写任务) [cite: 293]
    gamma: float = 1.0                 # 折扣因子
    lam: float = 0.95                  # GAE 参数
    
    # 生成参数
    response_length: int = 48          # 最大生成长度 [cite: 229]