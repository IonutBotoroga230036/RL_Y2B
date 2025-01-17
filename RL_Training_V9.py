import os
import wandb
import argparse
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ot2_env_wrapper_V4 import OT2Env  # Import the new env
from wandb.integration.sb3 import WandbCallback
import numpy as np

# Custom Callback to Save Best Model
class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Retrieve the latest training reward
        if "rollout/ep_rew_mean" in self.logger.records:
            mean_reward = self.logger.records["rollout/ep_rew_mean"][-1]
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"New best model saved with mean reward: {mean_reward}")
        return True

# Set CUDA device visibility if you do not have a CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0006, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for updating policy and value networks")
parser.add_argument("--n_steps", type=int, default=1024, help="Number of steps to run for each update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to update policy network")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for trade-off of bias vs variance in GAE")
parser.add_argument("--policy", type=str, default='MlpPolicy', help="Policy network architecture")
parser.add_argument("--hidden_units", type=int, default=32, help="Number of hidden units")
parser.add_argument("--total_timesteps", type=int, default=2500000, help="Total timesteps to train")

args = parser.parse_args()

# Initialize ClearML Task with arguments for logging
task = Task.init(project_name='Mentor Group S/Group 3',
                 task_name='RL_230036_W_V4_T_V8')
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

best_model_save_path = f"models/{run.id}/best_model"
save_best_model_callback = SaveBestModelCallback(save_path=best_model_save_path, verbose=1)

wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# Training Loop
time_steps = args.total_timesteps
for i in range(10):
    model.learn(
        total_timesteps=time_steps,
        callback=[wandb_callback, save_best_model_callback],
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}",
    )
    model.save(f"models/{run.id}/{time_steps*(i+1)}")