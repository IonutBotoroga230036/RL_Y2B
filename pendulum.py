from stable_baselines3 import PPO
import gym
import time
import os
import wandb
from wandb.integration.sb3 import WandbCallback


os.environ['WANDB_API_KEY'] = "4daea85f8e1f3362f5121c37348bc7da52586b7f"
env = gym.make('Pendulum-v1',g=9.81)

# initialize wandb project
run = wandb.init(project="sb3_pendulum_demo",sync_tensorboard=True)

# add tensorboard logging to the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}")

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# variable for how often to save the model
timesteps = 100000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{timesteps*(i+1)}")


# add wandb callback to the model training
model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, tb_log_name=f"runs/{run.id}")