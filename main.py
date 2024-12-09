import getopt
import json
import os
import sys
import json5

# 取消yolo打印信息
os.environ["YOLO_VERBOSE"] = str(False)

import pprint
import random
import pynput
import time
import cv2
import pydirectinput
import ctypes
from ultralytics import YOLO
import math
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import win32print
import heapq
import copy
from PIL import Image, ImageGrab

# import pyautogui 不要用这个库，cv2.imshow比例显示会有问题


# region 纯计算函数


def heuristic(node, goal):
    # 曼哈顿距离作为启发函数
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def a_star_search(start, goal, grid, road_block):
    open_set = [(heuristic(start, goal), start)]
    # print(open_set)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]
        # print(current)

        # 到达目标
        if current == goal:
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]  # pre current point
            path.insert(0, start)
            return path

        # 扩散
        diffuse = [
            (current[0] + 1, current[1]),  # 下
            (current[0] - 1, current[1]),  # 上
            (current[0], current[1] + 1),  # 右
            (current[0], current[1] - 1),  # 左
        ]
        if current in road_block:
            for e in road_block[current]:
                if e in diffuse:
                    diffuse.remove(e)
            # print(current, diffuse)

        for neighbor in diffuse:
            tentative_g_score = g_score[current] + 1
            if (
                0 <= neighbor[0] < len(grid)
                and 0 <= neighbor[1] < len(grid[0])
                and grid[neighbor[0]][neighbor[1]] == 0  # 0通行，1障碍
            ):
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if "\u4e00" <= ch <= "\u9fff":
            return True

    return False


def zhKey2key(zhpos):
    match zhpos:
        case "上":
            return "up"
        case "下":
            return "down"
        case "左":
            return "left"
        case "右":
            return "right"
        case "空" | "跳":
            return "space"
        case _:
            return zhpos


def is_similar_color(color1, color2, tolerance=10):
    """
    判断两个颜色是否相似

    Args:
        color1: 颜色1 (RGB)
        color2: 颜色2 (RGB)
        tolerance: 容差值，每个通道的差值小于该值则认为相似

    Returns:
        bool: 是否相似
    """
    return np.all(np.abs(np.array(color1) - np.array(color2)) <= tolerance)


def color_distance_hsv(color1, color2):
    """
    计算 HSV 空间中的颜色距离

    Args:
        color1: 颜色1 (BGR)
        color2: 颜色2 (BGR)

    Returns:
        float: 颜色距离
    """
    hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2HSV)[0][0]
    return np.linalg.norm(hsv1 - hsv2)


def count_pixel(img: cv2.typing.MatLike, colot_bgr, tolerance):
    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j]
            # if is_similar_color(color, img[i, j], tolerance=tolerance):
            if (
                math.fabs(x[0] - colot_bgr[0]) < tolerance
                and math.fabs(x[1] - colot_bgr[1]) < tolerance
                and math.fabs(x[2] - colot_bgr[2]) < tolerance
            ):
                count += 1
    return count


def formatted_boxes(boxes, names, type_index: int):
    boxes2 = []
    box_obj = {}
    for box in boxes:
        fb = [float(f"{nun:.2f}") for nun in box.tolist()]
        ti = names[int(fb[type_index])]
        if ti not in box_obj or box_obj[ti] is None:
            box_obj[ti] = []
        box_obj[ti].append(fb)
        boxes2.append(fb)
    return boxes2, box_obj


def rect_size(rect):
    (x1, y1, x2, y2) = rect[:4]
    return int(x2 - x1), int(y2 - y1)


def rect_center(rect):
    """计算矩形的中心点坐标"""
    (x1, y1, x2, y2) = rect[:4]
    return (x1 + x2) / 2, (y1 + y2) / 2


def calculate_distance(p_point, t_point):
    """计算两点之间的距离"""
    (x1, y1) = p_point
    (x2, y2) = t_point

    dx = x1 - x2
    dy = y1 - y2

    distance = math.sqrt(dx**2 + dy**2)

    return distance


def calculate_distance_and_angle(p_point, t_point):
    """计算两点之间的距离和角度"""
    (x1, y1) = p_point
    (x2, y2) = t_point

    math.atanh

    dx = x1 - x2
    dy = y1 - y2

    distance = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)  # 使用atan2函数可以更准确地计算角度，包括象限
    return distance, math.degrees(angle)


def find_nearest_target(player, target_list, min_distance):
    """寻找最近的target"""
    min_distance = float("inf")
    p_point = get_rect_point(player)

    nearest = None

    for target in target_list:
        t_point = get_rect_point(target)
        distance = calculate_distance(p_point, t_point)

        # 只要小于一定范围就可以返回了
        if distance < min_distance:
            return target

        if distance < min_distance:
            min_distance = distance
            nearest = target

    return nearest


def get_rect_point(rect):
    """获取矩形的某个点"""
    (x1, _, x2, y2) = rect[:4]
    x = (x1 + x2) / 2
    # center_y = (y1 + y2) / 2
    y = y2 * 0.99
    return int(x), int(y)


# endregion

# region class


class Skill(object):
    def __str__(self):
        return self.name if self.name else self.key

    def __init__(
        self,
        key,
        ct,
        mk: int | list[str] = None,
        buff=False,
        boss=False,
        charge=0,
        name="",
    ):
        """
        key: 键盘快捷键 或 上上下下z
        ct: 冷却时间
        mk: 多段技能, int重复按key多少次, list接下来要按的键位，只有快捷键可以
        buff: 状态技能
        boss: 这个技能只会在boss房间释放
        charge: 设置蓄力多少秒
        name: 技能名称
        """
        self.key = None
        self.lkey = None
        self.release_t = 0
        self.boss = boss
        self.charge = charge
        self.name = name
        self.buff = buff
        self.ct = ct

        # 如果包含中文则认定为指令
        if is_chinese(key):
            self.lkey = [zhKey2key(i) for i in key]
        else:
            self.key = key
        self.mk = mk
        # 如果位数字，则重复key
        if self.key and type(self.mk) == int:
            self.mk = [self.key for _ in range(self.mk)]

    def reset_ct(self):
        """重置技能冷却时间"""
        self.release_t = 0

    def can_release(self):
        return time.time() - self.release_t > self.ct

    def press_key(self, right=True):
        """
        right: 向右释放
        """

        if self.key:  # 有快捷键
            if self.charge > 0:
                key_down(self.key)
                time.sleep(self.charge)
                key_up(self.key)
            else:
                key_press(self.key)

            return True
        elif self.lkey:  # 操作指令
            lks_len = len(self.lkey) - 1
            for i, k in enumerate(self.lkey):
                if i == lks_len and self.charge > 0:
                    key_down(k)
                    time.sleep(self.charge)
                    key_up(k)
                else:
                    if not right:
                        match k:
                            case "left":
                                k = "right"
                            case "right":
                                k = "left"
                    key_press(k, _pause=False)
            return True

        return False

    def release(self, right=True):
        if not self.can_release():
            return False

        if not self.press_key(right):
            return False

        if self.key and self.mk:
            time.sleep(0.1)
            for k in self.mk:
                key_press(k)
                time.sleep(0.1)
        self.release_t = time.time()
        return True


class SkillBar(object):
    def __init__(self):
        self.list: list[Skill] = []

    def add(self, skill: Skill):
        self.list.append(skill)

    def add_many(self, skill_list: list[Skill]):
        for k in skill_list:
            self.add(k)

    def reset_ct_all(self):
        """重置所有技能冷却"""
        for k in self.list:
            k.reset_ct()

    # 释放所有buff技能
    def release_buff_all(self):
        for k in self.list:
            if k.buff and k.release():
                time.sleep(config["自动刷图"]["延迟"]["释放buff间隔"])

    # 释放攻击技能
    def release_attack(self, count=1, boss=False, right=True):
        """
        count 释放几个技能
        boss  释放终极技能
        """
        c = 0

        # 优先释放
        if boss:
            for k in self.list:
                if k.boss and k.release(right):
                    c += 1
                    if c >= count:
                        time.sleep(
                            config["自动刷图"]["延迟"]["释放终极技能"]
                        )  # 强制等待3秒技能动画
                        return c
                    time.sleep(0.2)

        for k in self.list:
            if k.buff or k.boss:
                continue
            if k.release(right):
                c += 1
                if c >= count:
                    break
                time.sleep(0.2)

        return c


class TestPlayerMoveSpeed:
    def __init__(self):
        """测试玩家移动速度，确保在大地图中测试"""
        self.from_point = None
        self.to_point = None

    def test(self, 玩家, keys, run):
        move_seconds = config["自动刷图"]["测试移速"]["秒"]
        p_point = get_rect_point(玩家)
        if not self.from_point:
            self.from_point = p_point
            key_down_many(keys, zhKey=True, run=run)
            time.sleep(move_seconds)
            key_up_many(keys, zhKey=True)
        elif self.from_point and not self.to_point:
            self.to_point = p_point
            distance = calculate_distance(self.from_point, self.to_point)
            ms = int(distance / move_seconds)
            print(f"移速:{ms}")
            return ms


class GameStatus(object):
    def __init__(self):
        self.swich_key = pynput.keyboard.Key.delete
        self.hwnd = None
        # 游戏窗口信息
        self.gw_info = {
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0,
            "w": 0,
            "h": 0,
            "scale": 1,
        }

        auto_cfg: dict = config["自动刷图"]
        # 一些固定的属性
        self._boss_bgr = tuple(auto_cfg["boss_bgr"])
        self._boss_tolerance = auto_cfg["boss_tolerance"]
        self._player_bgr = tuple(auto_cfg["player_bgr"])
        self._player_tolerance = auto_cfg["player_tolerance"]

        pi = auto_cfg["使用角色"]
        pinfo = config["角色列表"][pi]
        self.move_speed = tuple(pinfo["移速"])
        self.靠近放技能 = pinfo["靠近放技能"]

        self._room_road_block: dict = {}
        路线阻碍 = auto_cfg.get("路线阻碍")
        if 路线阻碍:
            for k in 路线阻碍:
                self._room_road_block[eval(k)] = [tuple(e) for e in 路线阻碍[k]]

        self._match_path = []  # 匹配路线
        匹配路线 = auto_cfg.get("匹配路线")
        if 匹配路线:
            for el in 匹配路线:
                self._match_path.append(
                    {
                        "begin": tuple(el["begin"]),
                        "end": tuple(el["end"]),
                        "path": [tuple(e) for e in el["path"]],
                    }
                )
        # 一些初始化的属性，在每次reset将初始化这些属性
        self._smap_grid: list[list[int]] = None

        # 游戏窗口内的矩形信息 (x1, y1, x2, y2)，这个矩形用来定位玩家在屏幕上的那个位置
        self.ss_rect = (0, 0, 0, 0)

        # 0可通行，1不可通行
        self.smap_grid = copy.deepcopy(self._smap_grid)
        # self.map_grid = [
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        # ]
        self.smap_item_size = auto_cfg["小地图每格大小"]
        self.smap_item_size_scal = 0
        self.smap_main_len = 0
        self.smap_cross_len = 0

        self.smap_points = None
        self.smap_items_rect = None
        self.room_i = 0
        self.to_boss_path_list: list | None = None
        self.room_road_block: dict[tuple, list[tuple]] = copy.deepcopy(
            self._room_road_block
        )
        self.next_room_info: tuple[str, tuple[int, int]] = None  # 下一个房间的位置信息

        self.boss_room_point_list = []  # 有可能有多个boss图标
        self.boss_room_point = None  # 最后boss
        self.player_room_point = None  # 玩家在小地图上的位置
        self.player_room_point_before = None  # 上一个房间的位置

        self.player_point = None
        self.player_point_before = None
        self.player_point_time = 0
        self.player_in_boss_room = False
        self.player_on_screen_pos_zh = ""
        self.场景 = ""  # 副本中
        self.疲劳值 = 100
        self.next_room_time = 0  # 过去了多少秒
        self.is_move_to_next_room = False
        self.next_room_is_boss = False

        # 再次挑战
        self.challenge_again = False

        # 走错了房间
        self.room_path_error = False

    def add_room_block(self):
        """路不通，意味着重新设计路线"""
        if not self.room_road_block:
            self.room_road_block = {}

        cur_room = self.player_room_point
        if cur_room not in self.room_road_block:
            self.room_road_block[cur_room] = []

        next_room_point = self.to_boss_path_list[self.room_i + 1]
        print(f"此路不通: {cur_room}到{next_room_point}")
        self.room_road_block[cur_room].append(next_room_point)

        self.to_boss_path_list = None  # 重置路径
        self.room_i = 0

    def reset_fb_status(self):
        self.challenge_again = True

        # 如果是固定地图，则不用重置
        self.smap_points = None

        # 这些都应该是必须初始化的
        self.to_boss_path_list = None
        self.room_i = 0
        self.场景 = ""
        self.boss_room_point_list.clear()
        self.boss_room_point = None
        self.player_room_point = None
        self.player_point = None
        self.player_point_before = None
        self.player_point_time = 0
        self.player_on_screen_pos_zh = ""
        self.next_room_info = None
        self.room_road_block = copy.deepcopy(self._room_road_block)
        self.player_in_boss_room = False

        self.smap_grid = copy.deepcopy(self._smap_grid)
        self.next_room_is_boss = False
        self.is_move_to_next_room = False

        skill_bar.reset_ct_all()

    def reset_path_status(self):
        self.to_boss_path_list = None
        self.player_room_point = None
        self.next_room_info = None
        self.room_i = 0


# endregion

# region global
config: dict = None
gs: GameStatus = None
yolo_model = None
testPlayerMoveSpeed = TestPlayerMoveSpeed()

boss_temp = cv2.imdecode(
    np.fromfile(r"./image_template/boss_icon.png", dtype=np.uint8),
    -1,
)
boss_temp_gray = cv2.cvtColor(boss_temp, cv2.COLOR_BGR2GRAY)

player_temp = cv2.imdecode(
    np.fromfile(r"./image_template/player_icon.png", dtype=np.uint8),
    -1,
)
player_temp_gray = cv2.cvtColor(player_temp, cv2.COLOR_BGR2GRAY)

user32 = ctypes.windll.user32
skill_bar = SkillBar()

# [x1, y1, x2, y2, 概率, 分类坐标]
box_param = {
    "x1": 0,
    "y1": 1,
    "x2": 2,
    "y2": 3,
    "概率": 4,
    "分类": 5,
}

COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}

random_moves_list = ["左上", "左下", "右上", "右下", "上", "右", "下", "左"]

# endregion


# region tools


def get_smap_move_zh_pos(current_node, next_node):
    """小地图房间坐标方向"""
    m1, c1 = current_node
    m2, c2 = next_node

    t_point = None
    x1, y1, x2, y2 = gs.ss_rect
    sw = gs.gw_info["w"]
    sh = gs.gw_info["h"]

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    if m2 > m1:
        pad = int(sw / 2)
        t_point = (xc, y2 + pad)
        return "下", t_point
    elif m2 < m1:
        pad = int(sw / 2)
        t_point = (xc, y1 - pad)
        return "上", t_point
    elif c2 > c1:
        pad = int(sh / 2)
        t_point = (x2 + pad, yc)
        return "右", t_point
    else:
        pad = int(sh / 2)
        t_point = (x1 - pad, yc)
        return "左", t_point


def get_ssr_pos_zh(point):
    x, y = point
    pos_zh = ""
    x1, y1, x2, y2 = gs.ss_rect

    if x <= x1:
        pos_zh += "左"
    if x >= x2:
        pos_zh += "右"
    if y <= y1:
        pos_zh += "上"
    if y >= y2:
        pos_zh += "下"
    return pos_zh


def degrees2PosZh(degrees, en=False):
    """将角度转换为方向"""
    # 右上 90 ~ 180
    # 右下 -90 ~ -180
    # 左上 0 ~ 90
    # 左下 0 ~ -90
    keys = ""
    if 0 < degrees < 90:
        # 左上
        keys = "LT" if en else "左上"
    elif -90 < degrees < 0:
        # 左下
        keys = "LB" if en else "左下"
    elif 90 < degrees < 180:
        # 右上
        keys = "RT" if en else "右上"
    elif -180 < degrees < -90:
        # 右下
        keys = "RB" if en else "右下"
    容差 = config["程序变量"]["degrees2PosZh"]["容差"]
    if (0 - 容差) < degrees < (0 + 容差):
        keys = "L" if en else "左"
    elif (90 - 容差) < degrees < (90 + 容差):
        keys = "T" if en else "上"
    elif ((180 - 容差) < degrees <= 180) or (-180 < degrees <= (-180 + 容差)):
        keys = "R" if en else "右"
    elif (-90 - 容差) < degrees < (-90 + 容差):
        keys = "B" if en else "下"

    return keys


def is_out_ssr(next_room_pos_zh: str, target, isPoint=False):
    """检查目标是否在ssr矩形外"""
    tx, ty = target if isPoint else get_rect_point(target)
    (x1, y1, x2, y2) = gs.ss_rect

    # 检测门是否在正确的边缘外
    if next_room_pos_zh == "上" and ty < y1:
        return True
    if next_room_pos_zh == "右" and tx > x2:
        return True
    if next_room_pos_zh == "下" and ty > y2:
        return True
    if next_room_pos_zh == "左" and tx < x1:
        return True

    return False


def cv2_cross_line(img, pos, size=10):
    x, y = pos
    cv2.line(img, (x - size, y), (x + size, y), COLORS["green"], thickness=1)
    cv2.line(img, (x, y - size), (x, y + size), COLORS["green"], thickness=1)


# 辅助线
def cv2_draw_auxiliary_line(img, p_point, t_point):
    # 交叉线
    cv2_cross_line(img, t_point)

    # 连接线
    cv2.line(img, p_point, t_point, COLORS["green"], thickness=1)

    # 距离
    distance, degrees = calculate_distance_and_angle(p_point, t_point)
    pos_zh = degrees2PosZh(degrees, en=True)
    cv2.putText(
        img,
        f"{pos_zh} {int(distance)} {int(degrees)}",
        (int((p_point[0] + t_point[0]) / 2), int((p_point[1] + t_point[1]) / 2)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=COLORS["red"],
        thickness=2,
    )


def key_press(key, _pause=True):
    # print(f"点击: {key}")
    pydirectinput.press(key, _pause=_pause)


# 记录所有按下的key
key_down_list = []


def key_down(key):
    if key not in key_down_list:
        # print(f"按下: {key}")
        key_down_list.append(key)
        pydirectinput.keyDown(key)


def key_up(key):
    # print(f"抬起: {key}")
    pydirectinput.keyUp(key)


def key_press_many(keys=None, zhKey=False):
    if type(keys) == str and zhKey:
        keys = list(map(lambda x: zhKey2key(x), list(keys)))

    for k in keys:
        key_press(k)
        time.sleep(0.1)


def key_up_many(keys=None, ignore=[], zhKey=False):
    if not keys or not len(keys):
        keys = key_down_list

    if type(keys) == str and zhKey:
        keys = list(map(lambda x: zhKey2key(x), list(keys)))

    for k in keys:
        if k in ignore:
            continue
        key_up(k)
        time.sleep(0.1)

    key_down_list.clear()


def key_down_many(keys, zhKey=False, run=False):
    if type(keys) == str and zhKey:
        keys = list(map(lambda x: zhKey2key(x), list(keys)))

    if run and len(keys) >= 1:
        key_press(keys[0], _pause=False)
        time.sleep(0.1)
        key_down_many(keys)
    else:
        for k in keys:
            key_down(k)
            time.sleep(0.1)


def random_move(logname, run=False):
    # print(f"随机游走: {logname}")
    zh_keys = random.choice(random_moves_list)
    key_down_many(zh_keys, zhKey=True, run=run)
    time.sleep(0.3)
    key_up_many()


def window_capture(hwnd, toCv2=False, usePIL=False):
    # 获取窗口尺寸
    rect = win32gui.GetWindowRect(hwnd)

    if usePIL:
        # 这会截取遮挡窗口
        game_frame = ImageGrab.grab(bbox=rect)
        return cv2.cvtColor(np.array(game_frame), cv2.COLOR_RGB2BGR)

    # 获取窗口的设备上下文DC
    hwndDC = win32gui.GetWindowDC(hwnd)

    real_w = win32print.GetDeviceCaps(hwndDC, win32con.DESKTOPHORZRES)
    apparent_w = win32api.GetSystemMetrics(0)
    scale = int(real_w / apparent_w)  # 计算出用户屏幕缩放了几倍

    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # rect = [item * scale for item in rect]

    left, top, right, bottom = rect
    w = right - left
    h = bottom - top

    gs.gw_info["left"] = left
    gs.gw_info["top"] = top
    gs.gw_info["right"] = right
    gs.gw_info["bottom"] = bottom
    gs.gw_info["w"] = w
    gs.gw_info["h"] = h
    gs.gw_info["scale"] = scale

    ssr_params = config["程序变量"]["ssr"]
    gs.ss_rect = (
        int(w * ssr_params[0]),
        int(h * ssr_params[1]),
        int(w * ssr_params[2]),
        int(h * ssr_params[3]),
    )

    # 创建位图对象
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    # 将设备上下文DC中的内容复制到位图对象中
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)

    # 保存图像
    signedIntsArray = saveBitMap.GetBitmapBits(True)
    img = Image.frombuffer("RGB", (w, h), signedIntsArray, "raw", "BGRX", 0, 1)
    # img.save("screenshot.png")

    # 释放对象
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    if toCv2:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def player_attack(player, target, x=False):
    # 使技能前，抬起所有键，避免卡住
    key_up_many()

    is_right = True
    if player and target:
        # 向哪个方向攻击
        px, _ = player if len(player) == 2 else get_rect_point(player)
        tx, _ = player if len(target) == 2 else get_rect_point(target)
        if tx > px:
            key_press("right")
            is_right = True
        else:
            key_press("left")
            is_right = False

    if x:
        k = config["自动刷图"]["快捷键"]["普攻"]
        key_down(k)
        time.sleep(config["自动刷图"]["延迟"]["按下普攻"])
        key_up(k)
    else:
        if (
            skill_bar.release_attack(1, boss=gs.player_in_boss_room, right=is_right)
            == 0
        ):
            k = config["自动刷图"]["快捷键"]["普攻"]
            key_down(k)
            time.sleep(config["自动刷图"]["延迟"]["按下普攻"])
            key_up(k)


def init_smap_items_rect_grid(small_map, pad: int):
    if gs.player_in_boss_room:
        return
    # 如果有了玩家和boss位置
    if gs.player_room_point and gs.boss_room_point:
        return

    if small_map:
        small_map = sorted(small_map, key=lambda x: x[box_param["概率"]])[-1]

    if not small_map:
        # print("没有识别到 副本地图")
        return

    gs.smap_points = []
    gs.smap_items_rect = []

    # 生成全图grid
    set_map_grid = False
    if not gs.smap_grid:
        gs.smap_grid = []
        set_map_grid = True

    w, h = rect_size(small_map)

    # 缩放后每隔格子的大小
    gs.smap_item_size_scal = int(gs.smap_item_size / gs.gw_info["scale"])

    gs.smap_main_len = int(round(h / gs.smap_item_size_scal))  # 横轴
    gs.smap_cross_len = int(round(w / gs.smap_item_size_scal))  # 纵轴

    small_map[box_param["x1"]] += pad
    small_map[box_param["y1"]] += pad

    x1 = small_map[box_param["x1"]]
    y1 = small_map[box_param["y1"]]

    for mi in range(0, gs.smap_main_len):
        gs.smap_points.append([])
        gs.smap_items_rect.append([])

        if set_map_grid:
            gs.smap_grid.append([])

        y2 = y1 + gs.smap_item_size_scal
        for ci in range(0, gs.smap_cross_len):
            x2 = x1 + gs.smap_item_size_scal

            gs.smap_items_rect[-1].append((int(x1), int(y1), int(x2), int(y2)))
            gs.smap_points[-1].append((mi, ci))

            if set_map_grid:
                gs.smap_grid[-1].append(0)

            x1 = x2

        x1 = small_map[box_param["x1"]]
        y1 = y2


# 设置player_room_point和boss_room_point
def smap_find_player_and_boss_point(img):
    if not gs.smap_items_rect or gs.player_in_boss_room:
        return

    player_locs: list[tuple[tuple[int, int], int]] = []
    boss_locs: list[tuple[tuple[int, int], int]] = []

    # 在小地图碎片中找到玩家和boss位置
    for mi in range(0, gs.smap_main_len):
        for ci in range(0, gs.smap_cross_len):
            smap_point = (mi, ci)
            x1, y1, x2, y2 = gs.smap_items_rect[mi][ci]

            # 切片
            smap_item_img = img[y1:y2, x1:x2]
            # cv2.imwrite(f'{mi}_{ci}.png',smap_item_img)

            if not gs.boss_room_point:
                boss_count = count_pixel(
                    smap_item_img, gs._boss_bgr, gs._boss_tolerance
                )
                if boss_count:
                    boss_locs.append((smap_point, boss_count))

            player_count = count_pixel(
                smap_item_img, gs._player_bgr, gs._player_tolerance
            )
            if player_count:
                player_locs.append((smap_point, player_count))

            continue

            # 缩放
            smap_item_img = cv2.resize(
                smap_item_img, (gs.smap_item_size, gs.smap_item_size)
            )
            # 灰度图像
            smap_item_img_gray = cv2.cvtColor(smap_item_img, cv2.COLOR_BGR2GRAY)

            # boss不会动，找一次够了
            if not gs.boss_room_point:
                boss_res = cv2.matchTemplate(
                    smap_item_img_gray, boss_temp_gray, cv2.TM_CCOEFF_NORMED
                )
                boss_loc = np.where(boss_res >= 0.8)
                if len(list(zip(*boss_loc[::-1]))):
                    boss_locs.append(
                        (
                            smap_point,
                            np.max(boss_res),
                            np.mean(smap_item_img_gray),
                        )
                    )

            # 玩家会移动房间
            player_res = cv2.matchTemplate(
                smap_item_img_gray, player_temp_gray, cv2.TM_CCOEFF_NORMED
            )
            player_loc = np.where(player_res >= 0.5)
            if len(list(zip(*player_loc[::-1]))):
                player_locs.append((smap_point, np.max(player_res)))

    # 进入副本
    if not gs.player_room_point and not gs.boss_room_point:
        skill_bar.release_buff_all()

    if len(player_locs):
        gs.player_room_point = sorted(player_locs, key=lambda x: x[1])[-1][0]
    else:
        if gs.next_room_is_boss:
            # BOSS房间是看不到玩家位置的
            gs.player_room_point = None
        elif (
            gs.to_boss_path_list
            and gs.room_i + 1 < len(gs.to_boss_path_list)
            and gs.is_move_to_next_room
        ):
            # 其他情况没看到则默认进入了下个房间，如果有进入房间的状态的话
            print("默认移动到下个房间")
            gs.player_room_point = gs.to_boss_path_list[gs.room_i + 1]
            gs.is_move_to_next_room = False

    if not gs.boss_room_point and len(boss_locs):
        gs.boss_room_point_list = boss_locs
        gs.boss_room_point = sorted(boss_locs, key=lambda x: x[1])[-1][0]

    # print(gs.player_room_point, gs.boss_room_point)


def find_door(next_room_pos_zh, door_list):
    i = 0
    j = 0

    match next_room_pos_zh:
        case "左" | "右":
            i = 0
        case "上" | "下":
            i = 1

    match next_room_pos_zh:
        case "左" | "上":  # 取最小
            j = 0
        case "右" | "下":  # 取最大
            j = -1

    door_point = sorted(
        map(lambda x: get_rect_point(x), door_list), key=lambda x: x[i]
    )[j]

    if is_out_ssr(next_room_pos_zh, door_point, isPoint=True):
        return door_point
    else:
        return None


def find_player_current_room_and_next_room():
    if gs.player_in_boss_room:
        return

    if gs.player_room_point and gs.to_boss_path_list:
        pre_room_i = gs.room_i
        try:
            room_i = gs.to_boss_path_list.index(gs.player_room_point)

            # 到达下个房间先加buff
            if room_i > pre_room_i:
                skill_bar.release_buff_all()

            gs.room_i = room_i

            # 获取下个房间位置
            next_room = (
                gs.to_boss_path_list[room_i]
                if gs.room_path_error
                else gs.to_boss_path_list[room_i + 1]
            )

            gs.next_room_is_boss = next_room == gs.boss_room_point
            gs.next_room_info = get_smap_move_zh_pos(gs.player_room_point, next_room)

            # print(f"下个房间位置: {gs.next_room_info[0]}")
        except:
            print("走错房间, 未在规定路线内")
            key_up_many()
            skill_bar.release_buff_all()
            gs.reset_path_status()
    else:
        if not gs.player_room_point:
            if gs.next_room_is_boss and not gs.player_in_boss_room:
                # print("BOSS房间")
                gs.player_in_boss_room = True
                if config["自动刷图"]["进BOSS房间直接释放技能"]:
                    skill_bar.release_attack(1, gs.player_in_boss_room)


def find_to_boss_room_path():
    # 寻找最佳路线
    if gs.player_room_point and gs.boss_room_point and not gs.to_boss_path_list:

        # 提供了匹配路径
        if gs._match_path:
            for el in gs._match_path:
                if (
                    el["begin"] == gs.player_room_point
                    and el["end"] == gs.boss_room_point
                ):
                    print(f'匹配到自定义路线 {el["begin"]}_{el["end"]}')
                    gs.to_boss_path_list = el["path"]
                    break

        # 使用a星寻路
        if not gs.to_boss_path_list:
            r = a_star_search(
                gs.player_room_point,
                gs.boss_room_point,
                gs.smap_grid,
                gs.room_road_block,
            )
            if not r:
                gs.room_road_block = {}
                r = a_star_search(
                    gs.player_room_point,
                    gs.boss_room_point,
                    gs.smap_grid,
                    gs.room_road_block,
                )

            gs.to_boss_path_list = r

        if gs.to_boss_path_list:
            print(
                f"{gs.smap_main_len}x{gs.smap_cross_len} 路线: {gs.to_boss_path_list} 玩家:{gs.player_room_point} BOSS:{gs.boss_room_point}"
            )


def handle_door_list(box_map: dict, player):
    """处理门列表"""
    if gs.player_in_boss_room:
        return

    door_list = box_map.get("门")
    if not door_list:
        return

    if not gs.player_room_point or not gs.boss_room_point:
        smap = box_map.get("副本地图")
        if not smap:
            return
        # 尽可能快的找到玩家在小地图上的位置
        for i in range(3):
            init_smap_items_rect_grid(smap, random.randint(-i, +i))
            if gs.player_room_point and gs.boss_room_point:
                break
            time.sleep(0.3)

    # 有下一个方向的信息
    if gs.next_room_info:
        next_room_pos_zh = gs.next_room_info[0]
        if next_room_pos_zh in gs.player_on_screen_pos_zh:
            gs.next_room_time = 0
            key_up_many()

            target_point = find_door(next_room_pos_zh, door_list)
            if target_point:
                # 前往下个房间前释放buff
                skill_bar.release_buff_all()
                params = config["自动刷图"]["移动方式"]["门"]
                move_to_target(
                    player,
                    target_point,
                    params["pad"],
                    run=params["run"],
                )
                gs.is_move_to_next_room = True
                key_up_many()
            else:
                print(
                    f"没找到门 从{gs.player_room_point}到{gs.to_boss_path_list[gs.room_i + 1]}"
                )

                # 提供了匹配路线不可能找不到门 或则当前路线以改变
                if not gs._match_path or gs.to_boss_path_list not in list(
                    map(lambda x: x["path"], gs._match_path)
                ):
                    gs.add_room_block()
                    return
        else:
            # 提供了匹配路线，觉表示绝对有路
            if not gs._match_path or gs.to_boss_path_list not in list(
                map(lambda x: x["path"], gs._match_path)
            ):
                if gs.next_room_time == 0:
                    print("设置1")
                    gs.next_room_time = time.time()

                # 说明这条路不通的
                过去了几秒 = int(time.time() - gs.next_room_time)
                if (
                    gs.next_room_time
                    and 过去了几秒 > config["自动刷图"]["延迟"]["路线卡住"]
                ):
                    ok = False
                    for door in door_list:
                        ok = is_out_ssr(next_room_pos_zh, door)
                        if ok:
                            gs.next_room_time = 0
                            break
                    print(f"还没有到{next_room_pos_zh}({过去了几秒})，匹配门:{ok}")
                    if not ok:
                        gs.next_room_time = 0
                        gs.add_room_block()
                        return
                    else:
                        random_move(
                            "前往下个房间",
                            run=config["自动刷图"]["移动方式"]["随机游走"]["run"],
                        )

            screen_point = gs.next_room_info[1]
            if screen_point:
                # move_to_target_xy(gs.player_point, screen_point, run=False)
                move_to_target(
                    gs.player_point,
                    screen_point,
                    run=config["自动刷图"]["移动方式"]["门"]["run"],
                    pad=5,
                )
            else:
                key_down_many(
                    next_room_pos_zh,
                    zhKey=True,
                    run=config["自动刷图"]["移动方式"]["门"]["run"],
                )


def move_to_target(
    player,
    target,
    pad,
    run=False,
):
    """移动到目标返回true"""
    p_point = player if len(player) == 2 else get_rect_point(player)
    t_point = target if len(target) == 2 else get_rect_point(target)

    distance, degrees = calculate_distance_and_angle(p_point, t_point)

    if distance <= pad:
        return True

    # 太近了就不跑了
    if distance < 200:
        run = False

    keys = degrees2PosZh(degrees)

    if keys:
        s_t = (distance - pad) / gs.move_speed[1 if keys == "上" or keys == "下" else 0]
        if run:
            s_t /= 2

        key_down_many(keys, zhKey=True, run=run)
        time.sleep(s_t)
        key_up_many()

    return False


def move_to_target_x(player_point, target_point, run=False):
    _, py = player_point
    tx, _ = target_point
    # 修补x
    player_point_to = (tx, py)
    move_to_target(
        player_point,
        player_point_to,
        1,
        run=run,
    )
    return player_point_to


def move_to_target_y(player_point, target_point, run=False):
    px, _ = player_point
    _, ty = target_point
    # 修补y
    player_point_to = (px, ty)
    move_to_target(
        player_point,
        player_point_to,
        1,
        run=run,
    )
    return player_point_to


def move_to_target_xy(player_point, target_point, run=False):
    """如果x打就修补x,否则修补y"""
    px, py = player_point
    tx, ty = target_point
    if math.fabs(tx - px) > math.fabs(ty - py):
        return move_to_target_x(player_point, target_point, run)
    else:
        return move_to_target_y(player_point, target_point, run)


def find_player(player_list):
    if player_list:
        # 获取概率最大的玩家
        player = sorted(player_list, key=lambda x: x[box_param["概率"]])[-1]

        # 在屏幕上的点位
        player_point = get_rect_point(player)

        if gs.player_point_before and player_point:
            # 检查两次的位置是否一样，如果被建筑卡住这很常见
            distance = calculate_distance(player_point, gs.player_point_before)
            if distance <= 5:
                if (
                    gs.player_point_time != 0
                    and time.time() - gs.player_point_time
                    > config["自动刷图"]["延迟"]["玩家卡住"]
                ):
                    key_up_many()
                    random_move(
                        "卡住了",
                        run=config["自动刷图"]["移动方式"]["随机游走"]["run"],
                    )
                return player

        gs.player_point_time = time.time()
        gs.player_point_before = gs.player_point
        gs.player_point = player_point

        # 处于屏幕哪个位置 上中下左？
        gs.player_on_screen_pos_zh = get_ssr_pos_zh(gs.player_point)

        return player


# endregion


# region main
def predict_source(source):
    """获取游戏屏幕，然后获取预测结果"""
    params = config["程序变量"]["predict"]
    result = yolo_model.predict(
        source=source,
        save=False,
        conf=params["conf"],
        device=params["device"],
    )[0]
    # print(result.names)
    img = np.array(result.plot())
    boxes = result.boxes.data
    # [2.2865e-01, 7.8152e+02, 2.2020e+02, 1.1498e+03, 9.8804e-01, 2.0000e+00],
    # pprint.pp(boxes)

    _, box_map = formatted_boxes(boxes, result.names, box_param["分类"])

    # [x1, y1, x2, y2, 概率, 分类坐标]
    # [1413.51, 880.17, 1516.58, 1162.73, 0.88, 1.0],
    # pprint.pp(boxes2)

    if config["调试窗口"]["标题"]:
        player_list = box_map.get("玩家")
        player = None

        if player_list:
            player = sorted(player_list, key=lambda x: x[box_param["概率"]])[-1]

        if player:
            p_point = get_rect_point(player)
            cv2_cross_line(img, p_point)

            敌人列表 = box_map.get("敌人")
            材料列表 = box_map.get("材料")
            门列表 = box_map.get("门")
            if 敌人列表:
                for e in 敌人列表:
                    cv2_draw_auxiliary_line(img, p_point, get_rect_point(e))
            if 材料列表:
                for e in 材料列表:
                    cv2_draw_auxiliary_line(img, p_point, get_rect_point(e))
            if 门列表:
                for e in 门列表:
                    cv2_draw_auxiliary_line(img, p_point, get_rect_point(e))

        cv2.rectangle(img, gs.ss_rect[:2], gs.ss_rect[2:], COLORS["green"], 2)

        if not gs.smap_points:
            init_smap_items_rect_grid(box_map.get("副本地图"), 0)

        if gs.smap_items_rect:
            color = (
                COLORS["red"]
                if not gs.player_room_point or not gs.boss_room_point
                else COLORS["green"]
            )
            for mi in range(0, gs.smap_main_len):
                for ci in range(0, gs.smap_cross_len):
                    rect = gs.smap_items_rect[mi][ci]
                    cv2.rectangle(
                        img,
                        rect[:2],
                        rect[2:],
                        (
                            COLORS["blue"]
                            if gs.to_boss_path_list and (mi, ci) in gs.to_boss_path_list
                            else color
                        ),
                        thickness=1,
                    )

    return img, box_map


def auto_game(box_map: dict, img):
    """处理yolo输出"""
    player = find_player(box_map.get("玩家"))

    测试移速 = config["自动刷图"]["测试移速"]
    if 测试移速["测试"]:
        if player:
            testPlayerMoveSpeed.test(player, 测试移速["方向"], 测试移速["run"])
        return False

    if box_map.get("选择界面"):
        gs.场景 = "选择界面"
    elif box_map.get("赛利亚"):
        gs.场景 = "赛利亚"
    elif box_map.get("返回城镇"):
        gs.场景 = "选择副本"
    else:
        gs.场景 = "副本中"

    if gs.场景 == "副本中":
        if (
            gs.challenge_again
            and not box_map.get("副本地图")
            and box_map.get("是否继续")
        ):
            print("可能没有疲劳了，返回城镇")
            key_press(config["自动刷图"]["快捷键"]["返回城镇"])
            time.sleep(1)

            # TODO: 点击esc，点击选择角色区域，选择下个角色
            # 停止程序
            return True

        enemy_list = box_map.get("敌人")
        materials_list = box_map.get("材料")
        door_list = box_map.get("门")

        if box_map.get("奖励"):
            key_up_many()
            # print("领取奖励")
            time.sleep(1)
            key_press(config["自动刷图"]["快捷键"]["esc"])
            time.sleep(config["自动刷图"]["延迟"]["领取奖励后"])

            key_press(config["自动刷图"]["快捷键"]["移动物品"])
            time.sleep(config["自动刷图"]["延迟"]["移动物品后"])

            # 自动修理武器
            key_press(config["自动刷图"]["快捷键"]["修理武器"])
            time.sleep(0.5)
            key_press(config["自动刷图"]["快捷键"]["确认"])
            return

        if box_map.get("是否继续"):
            key_up_many()
            # 初始化一些状态
            gs.reset_fb_status()
            # 继续挑战
            key_press(config["自动刷图"]["快捷键"]["再次挑战"])
            # 等会避免多余的动作
            time.sleep(config["自动刷图"]["延迟"]["再次挑战后"])
            return

        if player:
            # 获取小地图切片
            init_smap_items_rect_grid(box_map.get("副本地图"), 0)

            # 从小地图切片中找到玩家和boss定位
            smap_find_player_and_boss_point(img)

            # 从定位分析出前进路线
            find_to_boss_room_path()

            # 当前玩家在路线的哪个位置，以及下个路线的方向
            find_player_current_room_and_next_room()

            if enemy_list and not door_list:  # 有门就不打怪了
                key_up_many()
                gs.challenge_again = False

                # if gs.is_move_to_next_room:
                #     print("可能移动到了下一个房间")
                #     gs.is_move_to_next_room = False

                target = find_nearest_target(player, enemy_list, gs.靠近放技能)
                if target:
                    params = config["自动刷图"]["移动方式"]["打怪"]
                    player_point = get_rect_point(player)
                    target_point = get_rect_point(target)

                    if gs.靠近放技能 > 0:
                        # 站在目标的x轴上放技能
                        if target_point[0] > player_point[0]:
                            target_point = (
                                target_point[0] - gs.靠近放技能,
                                target_point[1],
                            )
                        else:
                            target_point = (
                                target_point[0] + gs.靠近放技能,
                                target_point[1],
                            )

                    player_point = move_to_target_y(
                        player_point, target_point, run=params["run"]
                    )
                    player_point = move_to_target_x(
                        player_point, target_point, run=params["run"]
                    )
                    ok = move_to_target(
                        player_point,
                        target_point,
                        params["pad"],
                        run=params["run"],
                    )
                    if config["自动刷图"]["移动后直接释放技能"] or ok:
                        player_attack(player_point, target)
                else:
                    print("没找到最近的敌人")

            elif materials_list and not gs.player_in_boss_room:
                key_up_many()

                target = find_nearest_target(player, materials_list, gs.靠近放技能)
                if target:
                    params = config["自动刷图"]["移动方式"]["捡材料"]
                    player_point = get_rect_point(player)
                    target_point = get_rect_point(target)

                    player_point = move_to_target_y(
                        player_point, target_point, run=params["run"]
                    )
                    player_point = move_to_target_x(
                        player_point, target_point, run=params["run"]
                    )

                    # move_to_target(
                    #     player_point,
                    #     target_point,
                    #     params["pad"],
                    #     run=params["run"],
                    # )
                    # key_up_many()
                    # key_press(config["自动刷图"]["快捷键"]["普攻"])
                else:
                    print("没找到最近的材料")

            elif door_list and not gs.player_in_boss_room:  # 在boss房间不管门
                key_up_many()

                # TODO：如果房间很大四周都没有门可能会出问题
                handle_door_list(box_map, player)
            else:
                # TODO: 判断之前有怪和材料，究竟可能向下个房间位置走
                random_move(
                    "除了玩家，其他什么都没有",
                    run=config["自动刷图"]["移动方式"]["随机游走"]["run"],
                )

        else:
            # 没找到玩家，但是有之前的point，则移动到ssr中间
            if gs.player_point_before:
                move_to_target(
                    gs.player_point_before,
                    rect_center(gs.ss_rect),
                    10,
                    run=False,
                )
                gs.player_point_before = None
            else:
                random_move(
                    "没有玩家", run=config["自动刷图"]["移动方式"]["随机游走"]["run"]
                )


def img_to_labelme_file(img, box_obj: dict, index: int, out_dir):
    out_file_name = f"{index}"
    out_img_ext_name = "jpg"
    out_img_filename = f"{out_file_name}.{out_img_ext_name}"
    if not cv2.imwrite(os.path.join(out_dir, out_img_filename), img):
        print("写入img文件失败")
        return False

    height, width, _ = img.shape

    labelme_file_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": out_img_filename,  # 替换为你的图片路径
        "imageData": None,
        "imageHeight": height,  # 替换为你的图片高度
        "imageWidth": width,  # 替换为你的图片宽度
    }

    for label in box_obj:
        val: list = box_obj[label]
        for rect in val:
            shape = {
                "label": label,  # 替换你的标签名
                "points": [rect[:2], rect[2:4]],  # 矩形
                "group_id": None,
                "shape_type": "rectangle",  # 类型 矩形
                "flags": {},
            }
            labelme_file_data["shapes"].append(shape)

    # 文件不存在会自动创建
    with open(
        os.path.join(out_dir, f"{out_file_name}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(labelme_file_data, f, indent=2)
        return True

    return False


def predict_game():
    if not yolo_model:
        print("没有TOLO模型")
        return

    调试窗口_标题 = config["调试窗口"]["标题"]
    if 调试窗口_标题:
        cv2.namedWindow(调试窗口_标题, cv2.WINDOW_NORMAL)
        wh = config["调试窗口"]["宽高"]
        if wh[0] > 0 and wh[1] > 0:
            cv2.resizeWindow(
                调试窗口_标题,
                wh[0],
                wh[1],
            )

    with pynput.keyboard.Listener(on_press=lambda key: not (key == gs.swich_key)) as hk:
        loop_i = 1
        mode = config.get("模式")
        if mode == "生成标记文件":
            out_dir = config["生成标记文件"]["输出目录"]
            try:
                os.makedirs(out_dir, exist_ok=False)
            except:
                print(f"目录已存在: {out_dir}")
                return

        in_img = None

        while hk.running:
            try:
                in_img = window_capture(gs.hwnd, toCv2=True)
            except:
                in_img = window_capture(gs.hwnd, toCv2=True, usePIL=True)

            # 游戏屏幕图片交给模型处理，得到输出的数据，输出的图片带有学习标记
            (img, box_obj) = predict_source(in_img)

            if mode == "生成标记文件":
                img_to_labelme_file(in_img, box_obj, loop_i, out_dir)
                loop_i += 1
                time.sleep(config["生成标记文件"]["seconds"])  # 避免生成太多文件

            if mode == "自动刷图":
                if auto_game(box_obj, in_img):
                    break
                if config["自动刷图"]["延迟"]["截图"] > 0:
                    time.sleep(config["自动刷图"]["延迟"]["截图"])

            # 显示一个测试窗口
            if 调试窗口_标题:
                cv2.imshow(调试窗口_标题, img)
                if config["调试窗口"]["置顶"]:
                    cv2.setWindowProperty(调试窗口_标题, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)

    cv2.destroyAllWindows()


def bootstrap():
    global yolo_model, config, gs
    opts, args = getopt.getopt(sys.argv[1:], "ho:", ["help", "output="])

    if len(args) == 0:
        print("缺少配置文件")
        return

    with open(args[0], "r", encoding="utf-8") as f:
        configData = f.read()
        i = configData.find("{")
        if i > 0:
            configData = configData[i:]
        config = json5.loads(configData)
        # pprint.pp(config)

        gs = GameStatus()

        mode = config.get("模式")
        yolo_model = YOLO(config["模型路径"])
        if mode == "自动刷图":
            pi = config[mode]["使用角色"]
            pinfo = config["角色列表"][pi]

            sk_list: list[dict] = config["技能表"][pinfo["技能"]]
            for obj in sk_list:
                sk = Skill(
                    key=obj.get("key", None),
                    ct=obj.get("ct"),
                    mk=obj.get("mk", None),
                    buff=obj.get("buff", False),
                    boss=obj.get("boss", False),
                    charge=obj.get("charge", 0),
                    name=obj.get("name", ""),
                )
                skill_bar.add(sk)

    with pynput.keyboard.Listener(on_press=lambda key: not (key == gs.swich_key)) as hk:
        while hk.running:
            print(f"选中游戏窗口，按{gs.swich_key}启动")
            time.sleep(1)

    gs.hwnd = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(gs.hwnd)
    if window_title.find(config["程序变量"]["游戏窗口标题"]) != -1:
        if config["模式"] == "截图模式":
            if not os.path.exists(config["截图模式"]["输出目录"]):
                print(f"输出目录不存在: {config['截图模式']['输出目录']}")
                return

            i = 1
            print(f"点击insert截屏, 点击:{gs.swich_key}退出")

            def on_press(key):
                nonlocal i
                if key == pynput.keyboard.Key.insert:
                    op = os.path.join(config["截图模式"]["输出目录"], f"{i}.jpg")
                    if cv2.imwrite(op, window_capture(gs.hwnd, toCv2=True)):
                        print(f"截图: {op}")
                        i += 1
                if key == gs.swich_key:
                    return False
                return True

            with pynput.keyboard.Listener(on_press=on_press) as hk:
                hk.join()
        else:
            predict_game()
    key_up_many()


# endregion

if __name__ == "__main__":
    bootstrap()
