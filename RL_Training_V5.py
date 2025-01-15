import os
import wandb
import argparse
from clearml import Task
from stable_baselines3 import PPO
from ot2_env_wrapper_V4 import OT2Env  # Import the new env
from wandb.integration.sb3 import WandbCallback
import numpy as np

# Set CUDA device visibility if you do not have a CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set up Argument Parser
parser = argparse.ArgumentParser(description='Train PPO agent with various hyperparameters')
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updating policy and value networks")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to update policy network")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for trade-off of bias vs variance in GAE")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter for PPO")
parser.add_argument("--policy", type=str, default='MlpPolicy', help="Policy network architecture")
parser.add_argument("--hidden_units", type=int, default=64, help="Number of hidden units")
parser.add_argument("--total_timesteps", type=int, default=500000, help="Total timesteps to train")
parser.add_argument("--eval_freq", type=int, default=20000, help="Frequency of evaluation")

args = parser.parse_args()

# Initialize ClearML Task with arguments for logging
task = Task.init(project_name='Mentor Group S/Group 3',
                 task_name='RL_230036')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.connect(args)  # logs arguments to clearml
task.execute_remotely(queue_name="default")  # removed to allow local execution

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--policy", type=str, default='MlpPolicy')

args = parser.parse_args()

env = OT2Env(render=False)

run = wandb.init(project="OT2_230036",sync_tensorboard=True)

model = PPO(args.policy, env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            gamma=args.gamma,
            tensorboard_log=f"runs/{run.id}",)

wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

time_steps = 200000
for i in range(10):
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    model.save(f"models/{run.id}/{time_steps*(i+1)}")