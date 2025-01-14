from stable_baselines3.common.env_checker import check_env
from ot2_env_wrapper import OT2Env

# instantiate your custom environment
wrapped_env = OT2Env() # modify this to match your wrapper class

# Assuming 'wrapped_env' is your wrapped environment instance
check_env(wrapped_env)
