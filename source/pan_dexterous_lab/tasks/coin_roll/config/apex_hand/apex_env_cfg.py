"""Re-export stage configs so gym entry points stay short."""

from pan_dexterous_lab.tasks.coin_roll.coin_roll_env_cfg import (
    BaodingRotateEnvCfg,
    BaodingRotateEnvCfg_PLAY,
    CoinHoldEnvCfg,
    CoinHoldEnvCfg_PLAY,
    CoinHoldLeftEnvCfg,
    CoinHoldVisionEnvCfg,
    CoinHoldVisionEnvCfg_PLAY,
    CoinTransferEnvCfg,
    CoinTransferEnvCfg_PLAY,
)

__all__ = [
    "BaodingRotateEnvCfg",
    "BaodingRotateEnvCfg_PLAY",
    "CoinHoldEnvCfg",
    "CoinHoldEnvCfg_PLAY",
    "CoinHoldLeftEnvCfg",
    "CoinHoldVisionEnvCfg",
    "CoinHoldVisionEnvCfg_PLAY",
    "CoinTransferEnvCfg",
    "CoinTransferEnvCfg_PLAY",
]
