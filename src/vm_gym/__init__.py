from gymnasium.envs.registration import register

register(
    id="VmEnv-1d",
    entry_point="src.vm_gym.envs.env1d:VmEnv"
)

register(
    id="VmEnv-2d",
    entry_point="src.vm_gym.envs.env2d:VmEnv"
)

register(
    id="VmEnv",
    entry_point="src.vm_gym.envs.env:VmEnv"
)