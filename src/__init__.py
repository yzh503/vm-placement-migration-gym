from gym.envs.registration import register

register(
    id='rlvm-v0',
    entry_point='src.envs:VmEnv',
)