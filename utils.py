# utils.py
import torch
import numpy as np
from transformers import pipeline

class RewardEngine:
    def __init__(self, model_name, device):
        # 初始化奖励模型 pipeline
        # 论文中使用了预训练好的分类器作为 Reward Model [cite: 238]
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model=model_name, 
            device=device,
            tokenizer=model_name
        )

    def get_rewards(self, texts):
        """
        计算 Reward (r)。
        为了简化，这里假设 label 'LABEL_2' (Positive) 是我们要的目标。
        """
        results = self.sentiment_pipe(texts, top_k=None)
        rewards = []
        for res in results:
            # 找到 positive 的分数
            score = next(item['score'] for item in res if item['label'] == 'LABEL_2')
            rewards.append(score)
        return torch.tensor(rewards)

def compute_intervention_weights(r_current, r_init):
    """
    核心代码：计算干预反馈权重 w
    对应论文公式 (1): w = |r_init - r| [cite: 154]
    """
    # 1. 计算绝对差值
    w_raw = torch.abs(r_init - r_current)
    
    # 2. Batch 内归一化 (Min-Max Normalization) [cite: 206]
    # 论文提到: "rescaled with min-max normalization denoted as w_hat"
    w_min = w_raw.min()
    w_max = w_raw.max()
    
    # 防止除以0
    if w_max - w_min < 1e-6:
        w_normalized = torch.ones_like(w_raw)  # 如果差异太小，则权重设为1
    else:
        w_normalized = (w_raw - w_min) / (w_max - w_min)
        
    return w_normalized