from ray import tune
import ray
from rocket_gym import RocketMeister10
import os
os.environ["TUNE_PLACEMENT_GROUP_AUTO_DISABLED"] = "0"
ray.init(local_mode=True)
tune.run(
    "SAC", # reinforced learning agent
    name = "SACt",
    # to resume training from a checkpoint, set the path accordingly:
    #resume = True, # you can resume from checkpoint
    #restore = r'D:\rocket-meister-master\rocket-meister-master\ray_results\ES\checkpoint_90\checkpoint-90',
    checkpoint_freq = 1,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": RocketMeister10,
        "num_workers": 30,
        #"num_gpus_per_worker": 0.1,
        "num_cpus_per_worker": 0.5,
        "env_config":{
            "max_steps": 1000,
            "export_frames": False,
            "export_states": False,
            "env_name" : "random",
            "camera_mode" : "centered",
            # "reward_mode": "continuous",
            # "env_flipped": True,
            # "env_flipmode": True,
            }
        },
    stop = {
        "timesteps_total": 50_000_000,
        },
    )