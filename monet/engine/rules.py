"""哨兵大战几何规则常量与纯函数(sentry_duel V4.4)。

这里只放"与状态无关"的规则:方向、镜像、视野、火力范围。
棋盘状态机在 game.py。
"""

from __future__ import annotations

BOARD_SIZE = 7
CELLS = BOARD_SIZE * BOARD_SIZE

# 出生点与初始朝向(绝对坐标)
SPAWN = {"R": (0, 0), "B": (6, 6)}
START_FACING = {"R": "E", "B": "W"}
OTHER = {"R": "B", "B": "R"}

# rules.md §2.5 示意图:障碍在 (1,1) 与 (5,5)
DEFAULT_OBSTACLES = ((1, 1), (5, 5))

# 中心十字 5 格
DEFAULT_SCORE_ZONES = ((3, 2), (2, 3), (3, 3), (4, 3), (3, 4))

# 方向:W 左 / E 右 / N 上 / S 下
DELTA = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIRS = ("N", "E", "S", "W")
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

# 垂直方向:用于 T 形视野与 3x3 火力范围的横向展开
PERP = {"N": (1, 0), "S": (1, 0), "E": (0, 1), "W": (0, 1)}

FIRE_RANGE = 3


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def mirror_pos(pos):
    """蓝方视角的 180 度镜像:(x, y) -> (6-x, 6-y)。"""
    return (BOARD_SIZE - 1 - pos[0], BOARD_SIZE - 1 - pos[1])


def mirror_dir(facing: str) -> str:
    if facing == "?":
        return "?"
    return OPPOSITE[facing]


# 镜像是对合(mirror_pos(mirror_pos(p)) == p),所以"绝对→视角"与"视角→绝对"
# 是**同一个函数**。下面两组名字只是给调用点读方向用的(game.py 说 to_view、
# 规则层说 to_absolute),不是两个不同的变换 —— 别照着其中一个去"修"另一个。
def to_absolute(color: str, pos):
    """把 color 视角下的坐标转成绝对坐标。"""
    return pos if color == "R" else mirror_pos(pos)


def to_view(color: str, pos):
    """把绝对坐标转成 color 视角下的坐标。"""
    return pos if color == "R" else mirror_pos(pos)


def view_dir(color: str, facing: str) -> str:
    return facing if color == "R" else mirror_dir(facing)


def abs_dir(color: str, facing: str) -> str:
    return facing if color == "R" else mirror_dir(facing)


def visible_cells(pos, facing, obstacles):
    """T 形视野(不含自己所在格),已扣除障碍遮挡。

    范围:正前方 1 格 + 距离 2 的横向 3 格。
    遮挡按格点独立判定(rules.md §5.3):
      - 距离 1 的格:自身是障碍则不可见;
      - 距离 2 的中心格:检查正前方中间格;
      - 距离 2 的左右侧格:检查对应的斜前中间格。
    """
    obstacles = obstacles if isinstance(obstacles, (set, frozenset)) else set(obstacles)
    fx, fy = DELTA[facing]
    px, py = PERP[facing]
    out = []

    x1, y1 = pos[0] + fx, pos[1] + fy
    if in_bounds(x1, y1) and (x1, y1) not in obstacles:
        out.append((x1, y1))

    # 距离 2 的三格**各自独立**判定遮挡(各自查自己的斜前/正前中间格),
    # 与正前方那一格可不可见无关 —— 正前方被挡不会连坐左右斜前。
    for lat in (-1, 0, 1):
        bx, by = x1 + lat * px, y1 + lat * py  # 斜前/正前中间格(= 遮挡判定格)
        cx, cy = pos[0] + 2 * fx + lat * px, pos[1] + 2 * fy + lat * py
        if not in_bounds(cx, cy) or (cx, cy) in obstacles:
            continue
        if not in_bounds(bx, by) or (bx, by) in obstacles:
            continue
        out.append((cx, cy))
    return out


def can_see(pos, facing, target, obstacles) -> bool:
    """target 是否在 pos/facing 的 T 形视野内(含遮挡判定)。"""
    if target == pos:
        return True
    obstacles = obstacles if isinstance(obstacles, (set, frozenset)) else set(obstacles)
    if target in obstacles:
        return False
    return target in set(visible_cells(pos, facing, obstacles))


def fire_hits(shooter_pos, facing, target, obstacles) -> bool:
    """3x3 火力范围内的命中判定(含障碍阻挡)。

    火力范围 = 朝向前方距离 1..3 的 3x3 区域。障碍会阻挡"同一火力通道"
    (即同一行/列平行射线)上障碍之后的格子。
    """
    obstacles = obstacles if isinstance(obstacles, (set, frozenset)) else set(obstacles)
    fx, fy = DELTA[facing]
    px, py = PERP[facing]
    for lat in (-1, 0, 1):
        sx, sy = shooter_pos[0] + lat * px, shooter_pos[1] + lat * py
        for step in range(1, FIRE_RANGE + 1):
            cx, cy = sx + step * fx, sy + step * fy
            if not in_bounds(cx, cy):
                break
            if (cx, cy) in obstacles:
                break
            if (cx, cy) == target:
                return True
    return False


def fire_cells(shooter_pos, facing, obstacles):
    """火力范围内所有可命中格(调试/可视化用)。"""
    obstacles = obstacles if isinstance(obstacles, (set, frozenset)) else set(obstacles)
    fx, fy = DELTA[facing]
    px, py = PERP[facing]
    cells = []
    for lat in (-1, 0, 1):
        sx, sy = shooter_pos[0] + lat * px, shooter_pos[1] + lat * py
        for step in range(1, FIRE_RANGE + 1):
            cx, cy = sx + step * fx, sy + step * fy
            if not in_bounds(cx, cy) or (cx, cy) in obstacles:
                break
            cells.append((cx, cy))
    return cells


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def best_turn_to_face(frm, to) -> str:
    """返回从 frm 指向 to 的主轴朝向(与 utils.h 的 best_turn_to_face 同语义)。"""
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else ("W" if dx < 0 else "S")
    return "S" if dy > 0 else "N"
