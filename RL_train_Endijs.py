import random
import os
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse
from sim_class import Simulation
from GOATED_ot2_env_wrapper import OT2Env
from clearml import Task
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

drop_command = 0


def linear_schedule(initial_value):
    """Linear learning rate schedule."""
    return lambda progress_remaining: progress_remaining * initial_value


def make_env(render=False, max_steps=1500):
    """Helper function to create monitored environments."""

    def _init():
        env = OT2Env(render=render, max_steps=max_steps)
        return Monitor(env)

    return _init

parser = argparse.ArgumentParser()
args = parser.parse_args()
# Initialize ClearML Task with arguments for logging
task = Task.init(project_name='Mentor Group S/Group 3',
                 task_name='RL_230036_Endijs_wrapper')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.connect(args)  # logs arguments to clearml
task.execute_remotely(queue_name="default")  # removed to allow local execution

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "4daea85f8e1f3362f5121c37348bc7da52586b7f"
    run = wandb.init(project="RL_OT2_Training", sync_tensorboard=True)

    # Configure WandbCallback
    wandb_callback = WandbCallback(
        model_save_freq=1000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    # Number of parallel environments
    num_envs = 16

    # Create vectorized training and evaluation environments
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(num_envs)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(render=False, max_steps=1500)]))

    # Select device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Define the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=linear_schedule(0.01),
        gamma=0.99,
        n_steps=1024,
        batch_size=128,
        vf_coef=0.4,
        clip_range=0.2,
        tensorboard_log=f"runs/{run.id}",
    )

    # Define evaluation callbacks
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=15, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,  # Only callback_on_best is passed here
        n_eval_episodes=100,
        eval_freq=50000,
        best_model_save_path="./best_model",  # Make sure this aligns with WandbCallback if needed
    )

    # Train the model
    model.learn(
        total_timesteps=5000000,
        callback=[wandb_callback],  # Both callbacks added here
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"PPO_{run.id}",
    )

    # Save the trained model
    model.save(f"models/ppo_ot2_{run.id}")
    print("Model training complete and saved.")
