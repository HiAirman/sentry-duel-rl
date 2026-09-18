"""与 sentry_duel.h 对齐的数据结构(选手视角)。

训练环境使用同一套名字,方便把选手 C++ 代码的观测逻辑 1:1 搬到 Python。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

Pos = Tuple[int, int]  # (x, y)

UNKNOWN_POS: Pos = (-1, -1)


@dataclass
class Sentry:
    """对应 struct Sentry。"""

    last_known_pos: Pos = UNKNOWN_POS
    last_known_facing: str = "?"
    visible: bool = False
    fire_cd: int = 0
    scan_cd: int = 0
    score: int = 0


@dataclass
class Board:
    """对应 struct Board —— 回合开始时的只读快照(已对调用方做镜像)。"""

    red: Sentry = field(default_factory=Sentry)
    blue: Sentry = field(default_factory=Sentry)
    turn: int = 0
    size: int = 7
    obstacles: List[Pos] = field(default_factory=list)
    score_zones: List[Pos] = field(default_factory=list)

    def me_opp(self, my_color: str):
        if my_color == "R":
            return self.red, self.blue
        return self.blue, self.red


@dataclass
class ActionObservation:
    """对应 struct ActionObservation。"""

    my_pos: Pos = UNKNOWN_POS
    my_facing: str = "?"
    opp_last_known_pos: Pos = UNKNOWN_POS
    opp_last_known_facing: str = "?"
    opp_visible: bool = False
    opp_directly_visible: bool = False
    fire_cd: int = 0
    scan_cd: int = 0


@dataclass
class ActionResult:
    """对应 struct ActionResult / ScanResult。"""

    success: bool = False
    consumed: bool = False
    observation: ActionObservation = field(default_factory=ActionObservation)
