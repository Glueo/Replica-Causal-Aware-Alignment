# main.py
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from config import CAAConfig
from utils import RewardEngine, compute_intervention_weights
from caa_trainer import CAAPPOTrainer # 假设使用了自定义 Trainer，或者使用原生 Trainer 手动改 Loss

# 1. 初始化配置
caa_config = CAAConfig()
ppo_config = PPOConfig(
    learning_rate=caa_config.learning_rate,
    batch_size=caa_config.batch_size,
    mini_batch_size=caa_config.mini_batch_size,
    ppo_epochs=caa_config.ppo_epochs,
)

# 2. 加载模型 [cite: 236]
# Policy Model (Current L)
model = AutoModelForCausalLMWithValueHead.from_pretrained(caa_config.model_name)
# Reference Model (Initial L_init)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(caa_config.ref_model_name)
tokenizer = AutoTokenizer.from_pretrained(caa_config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. 初始化 Trainer
# 这里使用 trl 的 PPOTrainer 来管理数据和生成，但我们需要手动干预 Loss
trainer = CAAPPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 4. 初始化奖励引擎
reward_engine = RewardEngine(caa_config.reward_model_name, trainer.accelerator.device)

# 5. 模拟数据加载器 (Prompt X)
# 实际使用时替换为 dataset loader
dummy_prompts = ["I think this movie is", "The food was", "I am 99% sure"] * 100 

print("Starting Causality-Aware Alignment Training...")

# --- 训练循环 ---
for epoch in tqdm(range(10)): # 示例 Epoch
    # 每次取一个 Batch
    for i in range(0, len(dummy_prompts), caa_config.batch_size):
        batch_prompts = dummy_prompts[i : i + caa_config.batch_size]
        
        # Encoding
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        query_tensors = inputs.input_ids.to(trainer.accelerator.device)
        
        # =================================================================
        # Step A: 生成 (Rollout) - 获取 y 和 y_init
        # =================================================================
        
        # 1. 当前模型生成 y [cite: 179]
        with torch.no_grad():
            response_tensors = trainer.generate(
                query_tensors, 
                max_new_tokens=caa_config.response_length,
                do_sample=True # 必须采样以获得多样性
            )
            batch_response_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # 2. 初始模型生成 y_init (为了计算 counterfactual baseline) [cite: 185]
        # 注意：这一步非常耗时，是 CAA 的计算开销所在
        with torch.no_grad():
            ref_response_tensors = trainer.generate(
                query_tensors,
                max_new_tokens=caa_config.response_length,
                do_sample=True,
                model=ref_model # 使用 ref_model 生成
            )
            batch_ref_response_text = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)

        # =================================================================
        # Step B: 奖励计算 (Reward) - 获取 r 和 r_init
        # =================================================================
        
        # 1. 计算当前生成的 Reward r [cite: 165]
        rewards_current = reward_engine.get_rewards(batch_response_text)
        
        # 2. 计算初始模型生成的 Reward r_init [cite: 204]
        rewards_init = reward_engine.get_rewards(batch_ref_response_text)
        
        # =================================================================
        # Step C: 计算因果干预权重 (Weights)
        # =================================================================
        
        # 计算归一化的权重 w_hat [cite: 205-206]
        # w = |r_init - r|, normalized
        intervention_weights = compute_intervention_weights(rewards_current, rewards_init)
        
        # Debug 打印
        if i == 0:
            print(f"Sample weights: {intervention_weights[:5]}")

        # =================================================================
        # Step D: PPO 更新 (带有加权 Loss)
        # =================================================================
        
        # 将 rewards 转换为 list 格式供 trainer 使用
        rewards_list = [r for r in rewards_current]
        
        # 运行 PPO Step
        # 注意：这里我们需要传入 intervention_weights 给修改后的 Trainer
        # 如果是标准 Trainer，这一步无法直接传入 weights，因此必须使用自定义的 CAAPPOTrainer
        stats = trainer.step(
            query_tensors, 
            response_tensors, 
            rewards_list,
            extra_args={"sample_weights": intervention_weights} # 假设修改后的 step 接受此参数
        )
        
        # 记录日志
        trainer.log_stats(stats, {"rewards": rewards_current}, args=None)

print("Training Finished!")