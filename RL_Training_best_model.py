import os
import wandb
import argparse
from clearml import Task
from stable_baselines3 import PPO
from ot2_env_wrapper_V2 import OT2Env
from wandb.integration.sb3 import WandbCallback
import numpy as np

# os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY" # Add your wandb API key or setup the environment
# Initialize ClearML Task
task = Task.init(project_name='Mentor Group S/Group 3',  # NB: Replace YourName with your own name
                 task_name='RL_best_model_230036')

task.set_base_docker('deanis/2023y2b-rl:latest')

task.execute_remotely(queue_name="default")

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
parser.add_argument("--threshold", type=float, default=0.001, help="Threshold for finishing the task")
parser.add_argument("--reward_distance_scale", type=int, default=100, help="Scale of the distance reward")
parser.add_argument("--step_penalty", type=float, default=-1, help="Penalty of each step taken")
parser.add_argument("--bonus_reward", type=int, default=100, help="Bonus given to the agent for finishing the task")
parser.add_argument("--total_timesteps", type=int, default=200000, help="Total timesteps to train")
parser.add_argument("--eval_freq", type=int, default=20000, help="Frequency of evaluation")



args = parser.parse_args()

# Initialize wandb
run = wandb.init(project="RL_OT2_V3", sync_tensorboard=True, config=vars(args))

# Initialize Environment
env = OT2Env(render=False, threshold=args.threshold,
             reward_distance_scale=args.reward_distance_scale, step_penalty=args.step_penalty,
             bonus_reward=args.bonus_reward)

# Initialize PPO model
model = PPO(args.policy, env,
            verbose=1,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            policy_kwargs=dict(net_arch=[args.hidden_units, args.hidden_units]),
            tensorboard_log=f"runs/{run.id}",
            )
# Initialize wandb callback
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )
# Training loop with early evaluation and saving
total_timesteps = args.total_timesteps
eval_freq = args.eval_freq

best_reward = -np.inf
best_model_path = None

for i in range(total_timesteps // eval_freq):
    model.learn(total_timesteps=eval_freq,
                callback=wandb_callback,
                progress_bar=True,
                reset_num_timesteps=False,
                tb_log_name=f"runs/{run.id}")
    model.save(f"models/{run.id}/{eval_freq*(i+1)}")
    
    # Evaluate the model on the training set
    rewards = []
    for i in range(5):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        while not terminated and not truncated:
          action, _ = model.predict(obs, deterministic=True)
          obs, reward, terminated, truncated, _ = env.step(action)
          episode_reward += reward
        rewards.append(episode_reward)

    avg_reward = np.mean(rewards)
    # Log to wandb
    wandb.log({"eval/avg_reward_training_set": avg_reward, "eval/std_reward_training_set": np.std(rewards)})
    print(f"Evaluation of training set: Avg reward {avg_reward:.2f} +/- {np.std(rewards):.2f}")

    # Save best model if new best reward is found
    if avg_reward > best_reward:
        best_reward = avg_reward
        if best_model_path is not None:
           os.remove(best_model_path) # Remove old best model
        best_model_path = f"models/{run.id}/best_model.zip"
        model.save(best_model_path)
        print(f"New best model saved at {best_model_path} with average reward {best_reward:.2f}")


run.finish()
env.close()