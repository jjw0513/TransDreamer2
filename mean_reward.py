import wandb
import numpy as np

# 로그인
wandb.login()

# 설정: entity와 project 지정
entity = 'hails'
project = '3ball_CAP'
runs = [
    #'3ball_DQN_5000_FullyObs',
    '3_ball_Dreamer_5000',
    '3ball_TransD_5000_2'
]

# 모델별 평균 보상 및 상위 5개 보상의 평균을 계산하는 함수
def calculate_rewards_stats(runs):
    api = wandb.Api()

    for run_id in runs:
        # 각 run의 데이터를 가져옴
        run = api.run(f"{entity}/{project}/{run_id}")

        # history로 로그 데이터 가져오기
        history = run.history(keys=["reward"])  # "reward" 필드를 가져옴
        rewards = history["reward"].to_numpy()

        # 평균 보상
        mean_reward = np.mean(rewards)

        # Top 5 보상
        top_5_mean = np.mean(np.sort(rewards)[-5:])

        print(f"Run: {run_id}")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Top 5 Mean Reward: {top_5_mean:.2f}")
        print("-" * 30)

# 실행
calculate_rewards_stats(runs)
