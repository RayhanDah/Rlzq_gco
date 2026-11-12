"""Bot RLBot v5 : adaptateur paquet RLBot -> `BatchState` et actions (ADR-004). Aucun import de RLBot ici."""

from rlbot_bot.adapter import (
    DEFAULT_TICK_SKIP,
    DOUBLE_JUMP_WINDOW_S,
    PacketAdapter,
    action_from_controls,
    controls_from_action,
    euler_to_quat,
)

__all__ = [
    "DEFAULT_TICK_SKIP",
    "DOUBLE_JUMP_WINDOW_S",
    "PacketAdapter",
    "action_from_controls",
    "controls_from_action",
    "euler_to_quat",
]
