from gymnasium.envs.registration import register

register(
    id="VmEnv-v1",
    entry_point="vmenv.envs.env:VmEnv"
)