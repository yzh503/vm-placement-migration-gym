from gymnasium.envs.registration import register

register(
    id="MultiDiscreteVmEnv-v1",
    entry_point="src.vm_gym.envs.mdenv:MultiDiscreteVmEnv"
)

register(
    id="VmEnv-v1",
    entry_point="src.vm_gym.envs.env:VmEnv"
)