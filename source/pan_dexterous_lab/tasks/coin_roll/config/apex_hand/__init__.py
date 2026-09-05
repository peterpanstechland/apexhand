import gymnasium as gym

from . import agents

gym.register(
    id="PAN-CoinHold-Apex-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:CoinHoldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CoinHoldPPORunnerCfg",
    },
)

gym.register(
    id="PAN-CoinHold-Apex-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:CoinHoldEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CoinHoldPPORunnerCfg",
    },
)

gym.register(
    id="PAN-CoinTransfer-Apex-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:CoinTransferEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CoinTransferPPORunnerCfg",
    },
)

gym.register(
    id="PAN-CoinTransfer-Apex-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:CoinTransferEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CoinTransferPPORunnerCfg",
    },
)

gym.register(
    id="PAN-CoinHold-Apex-Left-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "pan_dexterous_lab.tasks.coin_roll.coin_roll_env_cfg:CoinHoldLeftEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CoinHoldPPORunnerCfg",
    },
)

gym.register(
    id="PAN-BaodingRotate-Apex-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:BaodingRotateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaodingPPORunnerCfg",
    },
)

gym.register(
    id="PAN-BaodingRotate-Apex-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:BaodingRotateEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaodingPPORunnerCfg",
    },
)

gym.register(
    id="PAN-BaodingRotate-Apex-Left-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:BaodingRotateLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaodingPPORunnerCfg",
    },
)

gym.register(
    id="PAN-BaodingRotate-Apex-Left-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:BaodingRotateLeftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaodingPPORunnerCfg",
    },
)

gym.register(
    id="PAN-CoinHold-Apex-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:CoinHoldVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CoinHoldPPORunnerCfg",
    },
)

gym.register(
    id="PAN-CoinHold-Apex-Vision-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_env_cfg:CoinHoldVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CoinHoldPPORunnerCfg",
    },
)
