import os
import wandb
import argparse
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from ot2_env_wrapper_V5 import OT2Env  # Import the new env
from wandb.integration.sb3 import WandbCallback
import numpy as np

# Set CUDA device visibility if you do not have a CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0006, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for updating policy and value networks")
parser.add_argument("--n_steps", type=int, default=1024, help="Number of steps to run for each update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to update policy network")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for trade-off of bias vs variance in GAE")
parser.add_argument("--policy", type=str, default='MlpPolicy', help="Policy network architecture")
parser.add_argument("--hidden_units", type=int, default=32, help="Number of hidden units")
parser.add_argument("--total_timesteps", type=int, default=5000000, help="Total timesteps to train")
parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency of evaluation")
parser.add_argument("--n_eval_episodes", type=int, default=6, help="Number of evaluation episodes")

args = parser.parse_args()

# Initialize ClearML Task with arguments for logging
task = Task.init(project_name='Mentor Group S/Group 3',
                 task_name='RL_230036_W_V4_T_V7')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.connect(args)  # logs arguments to clearml
task.execute_remotely(queue_name="default")  # removed to allow local execution

env = OT2Env(render=False)

run = wandb.init(project="OT2_230036", sync_tensorboard=True)

model = PPO(args.policy, env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            policy_kwargs=dict(net_arch=[args.hidden_units, args.hidden_units]),
            tensorboard_log=f"runs/{run.id}",)

# Create an evaluation environment
eval_env = OT2Env(render=False)

# Add EvalCallback to monitor performance and save the best model
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=f"models/{run.id}/best_model",
                             log_path=f"models/{run.id}/logs",
                             eval_freq=args.eval_freq,
                             n_eval_episodes=args.n_eval_episodes,
                             deterministic=True,
                             verbose=1)

wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2)

time_steps = args.total_timesteps
for i in range(10):
    model.learn(total_timesteps=time_steps,
                callback=[wandb_callback, eval_callback],  # Include EvalCallback in the callback list
                progress_bar=True,
                reset_num_timesteps=False,
                tb_log_name=f"runs/{run.id}")
    model.save(f"models/{run.id}/{time_steps*(i+1)}")