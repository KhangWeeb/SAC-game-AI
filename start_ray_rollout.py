import gym, json
from ray.rllib import rollout
from ray.tune.registry import register_env


from rocket_gym import RocketMeister10

#ray.init(local_mode=True)
class MultiEnv(gym.Env):
    def __init__(self, env_config):
        self.env = RocketMeister10(env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
    def render(self, mode):
        return self.env.render(mode)
register_env("rocketmeister", lambda c: MultiEnv(c))

# path to checkpoint
checkpoint_path = r'D:\rocket-meister-master\rocket-meister-master\ray_results\checkpoint_49818\checkpoint-49818'

string = ' '.join([
    checkpoint_path,
    '--run',
    'SAC',
    '--env',
    'rocketmeister',
    '--episodes',
    '10',
    # '--no-render',
])

config = {
    'env_config': {
    #"export_frames": True,
    "export_states": False,
    'export_string': 'SAC', # filename prefix for exports
    'env_name' : 'random',
    'camera_mode' : 'centered',
    'env_random_length' : 200,
    #'env_flipmode' : True,
    #'env_flipped' : True,
    'gui_reward_total': True,
    'gui_echo_distances': False,
    'gui_level': False,
    'gui_velocity': False,
    'gui_goal_ang': False,
    'gui_frames_remaining': False,
    'gui_draw_echo_points': False,
    'gui_draw_echo_vectors': False,
    'gui_draw_goal_points': False,
    'gui_draw_goal_all' : False,
    'gui_draw_goal_next' : False,
    },
}
config_json = json.dumps(config)
parser = rollout.create_parser()
args = parser.parse_args(string.split() + ['--config', config_json])

# ──────────────────────────────────────────────────────────────────────────
# if you want to automate this, by calling rollout.run() multiple times, you
# uncomment the following lines too. They need to called before calling
# rollout.run() a second, third, etc. time
# ray.shutdown()
# tune.register_env("rocketgame", lambda c: MultiEnv(c))
# from ray.rllib import _register_all
# _register_all()
# ──────────────────────────────────────────────────────────────────────────

rollout.run(args, parser)