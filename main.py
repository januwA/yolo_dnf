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
from pynput.keyboard import Listener

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


def pos_reverse(keys: list[str] | str):
    """返回相反的方向"""
    if type(keys) == str:
        if is_chinese(keys):
            keys = list(keys)
        else:
            keys = [keys]

    res: list[str] = []

    for key in keys:
        if is_chinese(key):
            key = zhKey2key(key)

        match key:
            case "up":
                res.append("down")
            case "down":
                res.append("up")
            case "left":
                res.append("right")
            case "right":
                res.append("left")
            case _:
                res.append(key)
    return res


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


def count_pixel(img: cv2.typing.MatLike, color_bgr, tolerance):
    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j]
            # if is_similar_color(color, img[i, j], tolerance=tolerance):
            if (
                np.abs(x[0] - color_bgr[0]) < tolerance
                and np.abs(x[1] - color_bgr[1]) < tolerance
                and np.abs(x[2] - color_bgr[2]) < tolerance
            ):
                count += 1
    return count


def formatted_boxes(boxes, names, type_index: int):
    boxes2 = []
    box_map = {}
    for box in boxes:
        # fb = [float(f"{nun:.2f}") for nun in box.tolist()]
        fb = [int(nun) for nun in box.tolist()]
        ti = names[int(fb[type_index])]
        if ti not in box_map or box_map[ti] is None:
            box_map[ti] = []
        box_map[ti].append(fb)
        boxes2.append(fb)
    return boxes2, box_map


def rect_size(rect):
    (x1, y1, x2, y2) = rect[:4]
    return int(x2 - x1), int(y2 - y1)


def rect_center(rect):
    """计算矩形的中心点坐标"""
    (x1, y1, x2, y2) = rect[:4]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


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


def find_nearest_target(player, target_list, ok_distance):
    """寻找最近的target"""
    min_distance = float("inf")
    p_point = get_rect_point(player)

    nearest = None

    for target in target_list:
        t_point = get_rect_point(target)
        distance = calculate_distance(p_point, t_point)

        # 只要小于一定范围就可以返回了
        if distance < ok_distance:
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
            # print(f'释放技能: {self.key}')
            if self.charge > 0:
                key_down(self.key)
                time.sleep(self.charge)
                key_up(self.key)
            else:
                key_press(self.key)

            return True
        elif self.lkey:  # 操作指令
            # print(f'释放技能: {self.lkey}')
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
                time.sleep(0.5)

    # 释放攻击技能
    def release_attack(self, count=1, boss=False, right=True):
        c = 0

        # 优先释放
        if boss:
            for k in self.list:
                if k.boss and k.release(right):
                    c += 1
                    if c >= count:
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
        self.all = []
        self.from_point = None
        self.from_point_t = None
        self.to_point = None
        self.run = config["测试移速"]["run"]
        self.keys = config["测试移速"]["方向"]

    def test(self, 玩家):
        p_point = get_rect_point(玩家)
        if not self.from_point:
            self.from_point = p_point
            self.from_point_t = time.time()
            key_down_many(self.keys, run=self.run, seconds=0)

        elif self.from_point and not self.to_point:
            move_seconds = time.time() - self.from_point_t
            if move_seconds < config["测试移速"]["秒"]:
                return

            key_up_many(self.keys, seconds=0)
            self.to_point = p_point
            distance = calculate_distance(self.from_point, self.to_point)
            ms = int(distance / move_seconds)
            self.all.append(ms)
            均速 = sum(self.all) / len(self.all)
            print(f"移速:{ms}, {move_seconds:.2f}, 均速:{均速:.2f}")
            self.from_point = None
            self.to_point = None
            self.keys = pos_reverse(self.keys)
            return ms


class GameStatus(object):
    def __init__(self):
        self.pause = False
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
            "scale": 0,
        }

        cfg: dict = config["刷图模式"]

        # 游戏开始，城镇，副本
        self.场景 = cfg["场景"]

        # 一些固定的属性
        self._boss_bgr = tuple(cfg["boss_bgr"])
        self._boss_tolerance = cfg["boss_tolerance"]
        self._player_bgr = tuple(cfg["player_bgr"])
        self._player_tolerance = cfg["player_tolerance"]

        pi = cfg["使用角色"]
        pinfo = config["角色列表"][pi]
        self.move_speed = pinfo["移速"]
        self.释放距离 = pinfo["释放距离"]

        self._room_road_block: dict = {}
        road_block = cfg.get("路线阻碍")
        if road_block:
            for k in road_block:
                self._room_road_block[eval(k)] = [tuple(e) for e in road_block[k]]

        self._match_path = []  # 匹配路线
        match_path = cfg.get("匹配路线")
        if match_path:
            for el in match_path:
                self._match_path.append(
                    {
                        "begin": tuple(el["begin"]),
                        "end": tuple(el["end"]),
                        "path": [tuple(e) for e in el["path"]],
                    }
                )
        # 游戏窗口内的矩形信息 (x1, y1, x2, y2)，这个矩形用来定位玩家在屏幕上的那个位置
        self.ss_rect = (0, 0, 0, 0)

        # 0可通行，1不可通行
        self.smap_grid = None
        # self.map_grid = [
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        # ]
        self.room_i = 0
        self.room_path: list[tuple[int, int]] | None = None
        self.room_road_block: dict[tuple, list[tuple]] = copy.deepcopy(
            self._room_road_block
        )
        self.boss_room_point_list = []  # 有可能有多个boss图标
        self.boss_room_point = None  # 最后boss
        self.player_room_point = None  # 玩家在小地图上的位置
        self.player_room_point_before = None  # 上一个房间的位置

        self.player_point = None
        self.player_point_before = None
        self.player_point_time = 0
        self.player_in_boss_room = False
        self.next_room_time = 0  # 过去了多少秒
        self.is_move_to_next_room = False

        # 再次挑战
        self.challenge_again = False
        self.default_move_room = False

    def next_room_info(self):
        """从当前房间到下个房间的方位信息"""
        next_room_i = self.room_i + 1

        if (
            not gs.room_path
            or next_room_i >= len(self.room_path)
            or not self.player_room_point
        ):
            return None

        next_room = gs.room_path[next_room_i]
        return get_smap_move_zh_pos(self.player_room_point, next_room)

    def player_on_screen_pos_zh(self):
        """玩家处于屏幕上的哪个位置, 上中下左？"""
        if not self.player_point:
            return ""

        return get_ssr_pos_zh(self.player_point)

    def next_room_is_boss(self):
        """下个房间是boss"""
        next_room_i = self.room_i + 1
        if not self.room_path or next_room_i >= len(self.room_path):
            return False

        next_room = self.room_path[next_room_i]
        return next_room == self.boss_room_point

    def add_room_block(self, logname: str):
        """路不通，意味着重新设计路线"""
        if not self.room_road_block:
            self.room_road_block = {}

        cur_room = self.player_room_point
        if cur_room not in self.room_road_block:
            self.room_road_block[cur_room] = []

        next_room_point = self.room_path[self.room_i + 1]
        print(f"此路不通: {logname} {cur_room}到{next_room_point}")
        self.room_road_block[cur_room].append(next_room_point)

        self.room_path = None  # 重置路径
        self.room_i = 0

    def reset_fb_status(self):
        self.challenge_again = True

        # 这些都应该是必须初始化的
        self.room_path = None
        self.room_i = 0
        self.boss_room_point_list.clear()
        self.boss_room_point: tuple[int, int] = None
        self.player_room_point: tuple[int, int] = None
        self.player_point: tuple[int, int] = None
        self.player_point_before: tuple[int, int] = None
        self.player_point_time = 0
        self.room_road_block = copy.deepcopy(self._room_road_block)
        self.player_in_boss_room = False

        self.smap_grid = None
        self.is_move_to_next_room = False
        self.default_move_room = False

        skill_bar.reset_ct_all()

    def reset_path_status(self):
        self.room_path = None
        self.player_room_point = None
        self.room_i = 0


# endregion

# region global
# 原图
game_img: cv2.typing.MatLike = None
# 辅助线图
game_img2: cv2.typing.MatLike = None
# json导入的配置
config: dict = None
# 储存状态
gs: GameStatus = None
yolo_model = None
testPlayerMoveSpeed = None

# 小地图上boss图标
boss_icon_temp = None

# 小地图上玩家图标
player_icon_temp = None

# 地区传送
area_temp = None
area_temp_gray = None

# 选择地图时的副本
fb_temp = None
fb_temp_gray = None

# 选择角色按钮
select_role_temp = None
select_role_temp_gray = None

# boss特征
boss_feature_temp = None
boss_feature_temp_gray = None

# 城镇里的弹窗，只有点击关闭
close_button_temp = None
close_button_temp_gray = None

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


def load_json5(p: str):
    with open(p, "r", encoding="utf-8") as f:
        data = f.read()
        i = data.find("{")
        if i > 0:
            data = data[i:]
        return json5.loads(data)


def is_intersect(line, rect):
    """
    判断线段是否与矩形相交

    Args:
      line: 线段，表示为两个点的列表，如 [(0, 0), (100, 100)]
      rect: 矩形，表示为左下角和右上角坐标的列表，如 [(50, 50), (70, 70)]

    Returns:
      bool: True表示相交，False表示不相交
    """

    # 获取线段的两个端点和矩形的四个顶点
    x1, y1 = line[0]
    x2, y2 = line[1]
    rect_x1, rect_y1, rect_x2, rect_y2 = rect[:4]
    # rect_x1, rect_y1 = rect[0]
    # rect_x2, rect_y2 = rect[1]

    # 判断线段的两个端点是否在矩形内
    if (rect_x1 <= x1 <= rect_x2 and rect_y1 <= y1 <= rect_y2) or (
        rect_x1 <= x2 <= rect_x2 and rect_y1 <= y2 <= rect_y2
    ):
        return True

    # 判断线段是否与矩形的四条边相交
    # 这里使用向量叉积判断线段是否与矩形的边相交
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    # 矩形的四个顶点
    rect_points = [
        (rect_x1, rect_y1),
        (rect_x2, rect_y1),
        (rect_x2, rect_y2),
        (rect_x1, rect_y2),
    ]
    for i in range(4):
        if (
            ccw(line[0], line[1], rect_points[i])
            * ccw(line[0], line[1], rect_points[(i + 1) % 4])
            <= 0
            and ccw(rect_points[i], rect_points[(i + 1) % 4], line[0])
            * ccw(rect_points[i], rect_points[(i + 1) % 4], line[1])
            <= 0
        ):
            return True

    return False


def line_in_door(p1, p2, door_list):
    if door_list:
        line = [p1, p2]
        for door in door_list:
            res = is_intersect(line, door)
            if res:
                return True

    return False


def join_boss_room(logmsg):
    print(f"BOSS房间: {logmsg}")
    gs.player_in_boss_room = True
    if config["刷图模式"]["进BOSS房间直接释放技能"]:
        skill_bar.release_attack(1, gs.player_in_boss_room)


def load_img_and_gray(path: str, code=cv2.COLOR_BGR2GRAY):
    temp = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    temp_gray = None if code is None else cv2.cvtColor(temp, code)
    return temp, temp_gray


def load_image_temps():
    global area_temp, area_temp_gray, fb_temp, fb_temp_gray, select_role_temp, select_role_temp_gray, boss_feature_temp, boss_feature_temp_gray, boss_icon_temp, player_icon_temp, close_button_temp, close_button_temp_gray

    area_temp, area_temp_gray = load_img_and_gray(
        os.path.join(config["资源目录"], "地区传送.jpg")
    )
    fb_temp, fb_temp_gray = load_img_and_gray(
        os.path.join(config["资源目录"], "副本.jpg")
    )
    select_role_temp, select_role_temp_gray = load_img_and_gray(
        os.path.join(config["资源目录"], "选择角色.jpg")
    )
    boss_feature_temp, boss_feature_temp_gray = load_img_and_gray(
        os.path.join(config["资源目录"], "领主特征.jpg")
    )
    close_button_temp, close_button_temp_gray = load_img_and_gray(
        os.path.join(config["资源目录"], "关闭.jpg")
    )
    boss_icon_temp, _ = load_img_and_gray(
        os.path.join(config["资源目录"], "领主图标.jpg"), code=None
    )
    player_icon_temp, _ = load_img_and_gray(
        os.path.join(config["资源目录"], "玩家图标.jpg"), code=None
    )


def change_player():
    """切换角色，要更换技能，移速,释放距离"""
    skill_bar.list.clear()

    pi = config["刷图模式"]["使用角色"]
    pinfo = config["角色列表"][pi]
    gs.move_speed = pinfo["移速"]
    gs.释放距离 = pinfo["释放距离"]

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


def match_img_smap2(smap):
    """把每个房间切片在匹配"""
    w, h = rect_size(smap)

    smap_item_size = boss_icon_temp.shape[:2][-1]
    smap_main_len = int(round(h / smap_item_size))  # 横轴
    smap_cross_len = int(round(w / smap_item_size))  # 纵轴

    x1 = smap[box_param["x1"]]
    y1 = smap[box_param["y1"]]

    def m(img, temp):
        res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.5)
        if len(list(zip(*loc[::-1]))):
            _, _, _, maxVal = cv2.minMaxLoc(res)
            # 转化为整张图坐标
            maxVal = (x1 + maxVal[0], y1 + maxVal[1])
            return maxVal

        return None

    for mi in range(0, smap_main_len):
        y2 = y1 + smap_item_size
        for ci in range(0, smap_cross_len):
            x2 = x1 + smap_item_size
            smap_point = (mi, ci)

            smap_item_img = game_img[y1:y2, x1:x2]
            if m(smap_item_img, player_icon_temp):
                print(f"玩家:{smap_point}")
            if m(smap_item_img, boss_icon_temp):
                print(f"BOSS:{smap_point}")
            x1 = x2
        x1 = smap[box_param["x1"]]
        y1 = y2


def match_img_smap(smap, conf: float):
    """
    在副本小地图上找到玩家和boss
    使用彩色对比
    """
    x1, y1, x2, y2 = smap[:4]

    # 截取副本小地图
    fb_map = game_img[y1:y2, x1:x2]
    fb_map_gray = cv2.cvtColor(fb_map, cv2.COLOR_BGR2GRAY)

    def m(temp, gray=False):
        res = cv2.matchTemplate(
            fb_map_gray if gray else fb_map, temp, cv2.TM_CCOEFF_NORMED
        )
        loc = np.where(res >= conf)
        if len(list(zip(*loc[::-1]))):
            _, _, _, maxVal = cv2.minMaxLoc(res)
            # cv2.rectangle(fb_map, maxVal, (maxVal[0]+18,maxVal[1]+18), COLORS['red'], 2)
            # cv2.imwrite("fb_map2.jpg", fb_map)

            # 转化为整张图坐标
            maxVal = (x1 + maxVal[0], y1 + maxVal[1])
            return maxVal

        return None

    player_point = m(player_icon_temp)
    boss_point = None if gs.boss_room_point else m(boss_icon_temp)
    return player_point, boss_point


def match_img_click(temp_gray: cv2.typing.MatLike, conf: float, click=True, draw=False):
    """匹配图片"""
    global game_img, game_img2

    img_gray = cv2.cvtColor(game_img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
    temp_h, temp_w = temp_gray.shape[:2]
    locs = np.where(res >= conf)

    if len(list(zip(*locs[::-1]))):
        _, _, _, maxLoc = cv2.minMaxLoc(res)
        x, y = maxLoc
        rect = (x, y, x + temp_w, y + temp_h)

        # 屏幕坐标
        sx = gs.gw_info["left"] + x + int(temp_w / 2)
        sy = gs.gw_info["top"] + y + int(temp_h / 2)

        if draw:
            cv2.rectangle(game_img2, rect[:2], rect[2:], COLORS["red"], 2)
            # cv2.rectangle(game_img2, (sx, sy), (sx+50, sy+50), COLORS["blue"], 2)
        # mx, my = rect_center(rect)

        # 游戏图片坐标转换为屏幕坐标
        if click:
            sx = int(sx / config["程序变量"]["屏幕缩放"])
            sy = int(sy / config["程序变量"]["屏幕缩放"])
            mouse_click(sx, sy)
        return True

    return False


def gamestart_select_player(draw=False):
    """游戏开始选择角色"""
    player_i = config["刷图模式"]["使用角色"]
    col_len = config["刷图模式"]["游戏开始"]["列"]

    if draw:
        item_w = int(gs.gw_info["w"] / col_len)
        cv2.rectangle(game_img2, (0, 0), (item_w, item_w), COLORS["red"], 2)
        return

    print("重置选中角色")
    key_press_many("上" * 8)
    key_press_many("左" * col_len)

    print("选中角色")
    if player_i > 0:
        key_press_many("右" * player_i)

    print("确认选择")
    key_press(config["刷图模式"]["快捷键"]["确认"])


def get_smap_move_zh_pos(current_node, next_node):
    """小地图房间坐标方向"""
    m1, c1 = current_node
    m2, c2 = next_node

    t_point = None
    x1, y1, x2, y2 = gs.ss_rect
    sw = gs.gw_info["w"]
    sh = gs.gw_info["h"]

    xc = int((x1 + x2) / 2)
    yc = int((y1 + y2) / 2)

    xc = random.randint(xc - 50, xc + 50)
    yc = random.randint(yc - 50, yc + 50)

    if m2 > m1:
        pad = random.randint(int(sw / 2), sw)
        t_point = (xc, y2 + pad)
        return "下", t_point
    elif m2 < m1:
        pad = random.randint(int(sw / 2), sw)
        t_point = (xc, y1 - pad)
        return "上", t_point
    elif c2 > c1:
        pad = random.randint(int(sh / 2), sh)
        t_point = (x2 + pad, yc)
        return "右", t_point
    else:
        pad = random.randint(int(sh / 2), sh)
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
    x = config["程序变量"]["degrees2PosZh"]["容差"]
    if (0 - x) < degrees < (0 + x):
        keys = "L" if en else "左"
    elif (90 - x) < degrees < (90 + x):
        keys = "T" if en else "上"
    elif ((180 - x) < degrees <= 180) or (-180 < degrees <= (-180 + x)):
        keys = "R" if en else "右"
    elif (-90 - x) < degrees < (-90 + x):
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


def cv2_cross_line(pos, size=10):
    global game_img2

    x, y = pos
    cv2.line(game_img2, (x - size, y), (x + size, y), COLORS["green"], thickness=1)
    cv2.line(game_img2, (x, y - size), (x, y + size), COLORS["green"], thickness=1)


# 辅助线
def cv2_draw_auxiliary_line(p_point, t_point):
    global game_img2

    # 交叉线
    cv2_cross_line(t_point)

    # 连接线
    cv2.line(game_img2, p_point, t_point, COLORS["green"], thickness=1)

    # 距离
    distance, degrees = calculate_distance_and_angle(p_point, t_point)
    pos_zh = degrees2PosZh(degrees, en=True)
    cv2.putText(
        game_img2,
        f"{pos_zh} {int(distance)} {int(degrees)}",
        (int((p_point[0] + t_point[0]) / 2), int((p_point[1] + t_point[1]) / 2)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=COLORS["red"],
        thickness=2,
    )


def mouse_down(sx, sy):
    pydirectinput.mouseDown(sx, sy)


def mouse_up(sx, sy):
    pydirectinput.mouseUp(sx, sy)


def mouse_click(sx, sy):
    mouse_down(sx, sy)
    time.sleep(0.3)
    mouse_up(sx, sy)


def key_press(key, _pause=True):
    if is_chinese(key):
        key = zhKey2key(key)

    # print(f"点击: {key}")
    pydirectinput.press(key, _pause=_pause)


# 记录所有按下的key
key_down_list = []


def key_down(key):
    if is_chinese(key):
        key = zhKey2key(key)

    if key not in key_down_list:
        # print(f"按下: {key}")
        key_down_list.append(key)
        pydirectinput.keyDown(key)


def key_up(key):
    if is_chinese(key):
        key = zhKey2key(key)

    # print(f"抬起: {key}")
    pydirectinput.keyUp(key)


def key_press_many(keys=None):
    if type(keys) == str and is_chinese(keys):
        keys = list(map(lambda x: zhKey2key(x), list(keys)))

    for k in keys:
        key_press(k)
        time.sleep(0.1)


def key_up_many(keys=None, ignore=[], seconds=0.1):
    if not keys or not len(keys):
        keys = key_down_list

    if type(keys) == str and is_chinese(keys):
        keys = list(map(lambda x: zhKey2key(x), list(keys)))

    for k in keys:
        if k in ignore:
            continue
        key_up(k)
        time.sleep(seconds)

    key_down_list.clear()


def key_down_many(keys: list[str] | str, run=False, seconds=0.1):
    if type(keys) == str and is_chinese(keys):
        keys = list(map(lambda x: zhKey2key(x), list(keys)))

    if run and len(keys) >= 1:
        key_press(keys[0], _pause=False)
        # time.sleep(seconds)
        key_down_many(keys, run=False, seconds=seconds)
    else:
        for k in keys:
            key_down(k)
            time.sleep(seconds)


before_random_pos = None


def random_move(logname, run=False, seconds=0.2):
    global before_random_pos

    zh_keys = None
    while True:
        zh_keys = random.choice(random_moves_list)
        if before_random_pos == zh_keys:
            continue
        break

    before_random_pos = zh_keys
    print(f"随机游走({zh_keys}): {logname}")
    key_down_many(zh_keys, run=run)
    time.sleep(seconds)
    key_up_many()


def window_capture(hwnd, toCv2=False, usePIL=False):
    hwndDC = win32gui.GetWindowDC(hwnd)

    # 获取窗口的设备上下文DC
    if gs.gw_info["scale"] == 0:
        real_w = win32print.GetDeviceCaps(hwndDC, win32con.DESKTOPHORZRES)
        apparent_w = win32api.GetSystemMetrics(0)
        gs.gw_info["scale"] = int(real_w / apparent_w)  # 计算出用户屏幕缩放了几倍
        if config["程序变量"]["屏幕缩放"] < 0:
            config["程序变量"]["屏幕缩放"] = gs.gw_info["scale"]

    # 获取窗口尺寸
    rect = win32gui.GetWindowRect(hwnd)
    rect = [int(s * config["程序变量"]["屏幕缩放"]) for s in rect]
    left, top, right, bottom = rect
    w = right - left
    h = bottom - top
    gs.gw_info["left"] = left
    gs.gw_info["top"] = top
    gs.gw_info["right"] = right
    gs.gw_info["bottom"] = bottom
    gs.gw_info["w"] = w
    gs.gw_info["h"] = h
    ssr_params = config["程序变量"]["ssr"]
    gs.ss_rect = (
        int(w * ssr_params[0]),
        int(h * ssr_params[1]),
        int(w * ssr_params[2]),
        int(h * ssr_params[3]),
    )

    if usePIL:
        win32gui.ReleaseDC(hwnd, hwndDC)
        # 这会截取遮挡窗口
        game_frame = ImageGrab.grab(bbox=rect)
        return cv2.cvtColor(np.array(game_frame), cv2.COLOR_RGB2BGR)

    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

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
        k = config["刷图模式"]["快捷键"]["普攻"]
        key_down(k)
        time.sleep(config["刷图模式"]["延迟"]["按下普攻"])
        key_up(k)
    else:
        if (
            skill_bar.release_attack(1, boss=gs.player_in_boss_room, right=is_right)
            == 0
        ):
            k = config["刷图模式"]["快捷键"]["普攻"]
            key_down(k)
            time.sleep(config["刷图模式"]["延迟"]["按下普攻"])
            key_up(k)


def match_smap_room_loc(smap):
    if gs.player_in_boss_room or not smap:
        return

    if smap:
        smap = sorted(smap, key=lambda x: x[box_param["概率"]])[-1]

    w, h = rect_size(smap)
    smap_item_width = boss_icon_temp.shape[:2][-1]  # 用boss图标的宽度来预测每隔宽度
    smap_main_len = int(round(h / smap_item_width))  # 横轴
    smap_cross_len = int(round(w / smap_item_width))  # 纵轴
    # print(f'{smap_main_len}x{smap_cross_len}')

    # match_img_smap2(smap)
    # return

    # 进入副本
    if not gs.player_room_point and not gs.boss_room_point:
        skill_bar.release_buff_all()

    player_point, boss_point = match_img_smap(
        smap, config["程序变量"]["置信度"]["副本小地图"]
    )

    # 生成全图grid，只生成一次
    not_smap_grid = gs.smap_grid is None
    if not_smap_grid:
        gs.smap_grid = []

    x1 = smap[box_param["x1"]]
    y1 = smap[box_param["y1"]]
    is_find_player = False

    for mi in range(0, smap_main_len):
        if not_smap_grid:
            gs.smap_grid.append([])
        y2 = y1 + smap_item_width
        for ci in range(0, smap_cross_len):
            x2 = x1 + smap_item_width
            smap_point = (mi, ci)

            if player_point and not is_find_player:
                if x1 < player_point[0] < x2 and y1 < player_point[1] < y2:
                    is_find_player = True
                    gs.player_room_point = smap_point

            # 只找一次boss
            if boss_point and not gs.boss_room_point:
                if x1 < boss_point[0] < x2 and y1 < boss_point[1] < y2:
                    gs.boss_room_point = smap_point

            # 使用像素查找
            # if not gs.player_room_point or not gs.boss_room_point:
            #     smap_item_img = game_img[y1:y2, x1:x2]
            #     # cv2.imwrite(f'{mi}_{ci}.jpg',smap_item_img)

            #     if not gs.boss_room_point:
            #         boss_count = count_pixel(
            #             smap_item_img, gs._boss_bgr, gs._boss_tolerance
            #         )
            #         print(f'{boss_count=}')
            #         if boss_count:
            #             gs.boss_room_point = smap_point

            #     if not find_player:
            #         player_count = count_pixel(
            #             smap_item_img, gs._player_bgr, gs._player_tolerance
            #         )
            #         print(f'{player_count=}')
            #         if player_count:
            #             find_player = True
            #             gs.player_room_point = smap_point
            #             print("查找到到玩家")

            if not_smap_grid:
                gs.smap_grid[-1].append(0)

            x1 = x2

        x1 = smap[box_param["x1"]]
        y1 = y2

    if not is_find_player:
        if gs.next_room_is_boss():
            # BOSS房间是看不到玩家位置的
            if gs.is_move_to_next_room:
                gs.player_room_point = None
        elif (
            gs.room_path
            and gs.room_i + 1 < len(gs.room_path)
            and gs.is_move_to_next_room
            and not gs.default_move_room
        ):
            gs.default_move_room = True  # 一个房间之移动一次
            gs.player_room_point = gs.room_path[gs.room_i + 1]
            print(f"默认移动到下个房间 {gs.player_room_point}")
            gs.is_move_to_next_room = False


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

    if gs.player_room_point and gs.room_path:
        pre_room_i = gs.room_i
        try:
            room_i = gs.room_path.index(gs.player_room_point)

            # 到达下个房间先加buff
            # if room_i > pre_room_i:
            # print(f"房间已到达 {gs.player_room_point}")
            # skill_bar.release_buff_all()

            gs.room_i = room_i
        except:
            # skill_bar.release_buff_all()
            当前房间 = gs.player_room_point
            上个房间 = gs.room_path[gs.room_i]
            print(f"走错房间 当前房间:{当前房间} 上个房间:{上个房间} {gs.room_i=}")
            # 切掉走过的路线
            gs.room_path = gs.room_path[gs.room_i :]
            # 将当前房间添加到最前面
            gs.room_path.insert(0, 当前房间)
            print(f"新路线:{gs.room_path}")
            # g = [
            #     (0, 0), (0, 1), (0, 2),
            #     (1, 0), (1, 1), (1, 2),
            #     (2, 0), (2, 1), (2, 2),
            # ]
            # p = [(0, 0), (0, 1), (0, 2)]
            # # p = [(0,0), (0,1), (1,1), (0,1) (0,2)]
            # # p = [(0,0), (0,1), (1,1), (2,1), (1,1), (0,1) (0,2)]
            # p.insert(1 + 1, (0, 1))
            # p.insert(1 + 1, (1, 1))

            # p.insert(2 + 1, (1, 1))
            # p.insert(2 + 1, (2, 1))
            # print(p)
            return False
            # key_up_many()
            # gs.reset_path_status()
    else:
        if not gs.player_room_point:
            if gs.next_room_is_boss() and not gs.player_in_boss_room:
                join_boss_room("房间判断")


def find_to_boss_room_path():
    # 寻找最佳路线
    if gs.player_room_point and gs.boss_room_point and not gs.room_path:

        # 提供了匹配路径
        if gs._match_path:
            for el in gs._match_path:
                if (
                    el["begin"] == gs.player_room_point
                    and el["end"] == gs.boss_room_point
                ):
                    # print(f'匹配到自定义路线 {el["begin"]}_{el["end"]}')
                    gs.room_path = el["path"]
                    break

        # 使用a星寻路
        if not gs.room_path:
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

            gs.room_path = r


def handle_door_list(box_map: dict, player):
    """处理门列表"""
    if gs.player_in_boss_room:
        return

    door_list = box_map.get(config["label"]["men"])
    if not door_list:
        return

    gs.is_move_to_next_room = True

    # 有下一个方向的信息
    nrf = gs.next_room_info()
    if nrf:
        next_room_pos_zh = nrf[0]
        if next_room_pos_zh in gs.player_on_screen_pos_zh():
            gs.next_room_time = 0
            key_up_many()

            target_point = find_door(next_room_pos_zh, door_list)
            if target_point:
                # 前往下个房间前释放buff
                # skill_bar.release_buff_all()
                player_point = get_rect_point(player)
                distance = calculate_distance(player_point, target_point)
                params = {
                    "pad": random.randint(1, 5),
                    "run": distance > gs.gw_info["w"] / 3,
                }
                move_to_target(
                    "门",
                    player_point,
                    target_point,
                    params["pad"],
                    run=params["run"],
                )
            else:
                # print(f"没找到门 从{gs.player_room_point}到{gs.to_boss_path_list[gs.room_i + 1]}")
                # 提供了匹配路线不可能找不到门 或则当前路线以改变
                if not gs._match_path or gs.room_path not in list(
                    map(lambda x: x["path"], gs._match_path)
                ):
                    gs.add_room_block(
                        f"到了{gs.player_on_screen_pos_zh()},没有门({len(door_list)})"
                    )
                    return
                else:
                    random_move("没找到门")
        else:
            # 提供了匹配路线，觉表示绝对有路
            if not gs._match_path or gs.room_path not in list(
                map(lambda x: x["path"], gs._match_path)
            ):
                if gs.next_room_time == 0:
                    gs.next_room_time = time.time()

                # 说明这条路不通的
                过去了几秒 = int(time.time() - gs.next_room_time)
                if (
                    gs.next_room_time
                    and 过去了几秒 > config["刷图模式"]["延迟"]["路线卡住"]
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
                        gs.add_room_block(f"过去了{int(过去了几秒)}秒，还没找到门")
                        return
                    else:
                        random_move("前往下个房间")

            screen_point = nrf[1]
            if screen_point:
                # move_to_target_xy(gs.player_point, screen_point, run=False)
                move_to_target(
                    f"下个房间方向({nrf[0]})",
                    gs.player_point,
                    screen_point,
                    run=True,
                    pad=random.randint(5, 10),
                )
            else:
                key_down_many(next_room_pos_zh, run=True)


def move_to_target(logname: str, player, target, pad, run=False):
    """移动到目标返回true"""
    # print(f"移动到: {logname}")
    p_point = player if len(player) == 2 else get_rect_point(player)
    t_point = target if len(target) == 2 else get_rect_point(target)

    distance, degrees = calculate_distance_and_angle(p_point, t_point)

    if distance <= pad:
        return True

    # 太近了就不跑了
    if distance < random.randint(80, 120):
        run = False

    keys = degrees2PosZh(degrees)

    # print(keys)

    if keys:
        s_t = (distance - pad) / gs.move_speed
        if run:
            s_t /= 2

        key_down_many(keys, run=run)
        time.sleep(s_t)
        key_up_many()

    return False


def patch_point_x(p1, p2):
    """移动p1的x到p2"""
    _, py = p1
    tx, _ = p2
    return (tx, py)


def patch_point_y(p1, p2):
    """移动p1的y到p2"""
    px, _ = p1
    _, ty = p2
    return (px, ty)


def move_to_target_x(logname: str, player_point, target_point, run=False):
    player_point_to = patch_point_x(player_point, target_point)
    move_to_target(
        logname,
        player_point,
        player_point_to,
        1,
        run=run,
    )
    return player_point_to


def move_to_target_y(logname: str, player_point, target_point, run=False):
    player_point_to = patch_point_y(player_point, target_point)
    move_to_target(
        logname,
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
    if np.abs(tx - px) > np.abs(ty - py):
        return move_to_target_x(player_point, target_point, run)
    else:
        return move_to_target_y(player_point, target_point, run)


def find_player(box_map):
    player_list = box_map.get(config["label"]["wanjia"])
    if player_list:
        # 获取概率最大的玩家
        player = sorted(player_list, key=lambda x: x[box_param["概率"]])[-1]

        # 在屏幕上的点位
        player_point = get_rect_point(player)

        if gs.player_point_before and player_point:
            # 检查两次的位置是否一样，如果被建筑卡住这很常见
            distance = calculate_distance(player_point, gs.player_point_before)
            if distance <= random.randint(4, 6):
                s = time.time() - gs.player_point_time
                if (
                    gs.player_point_time != 0
                    and s > config["刷图模式"]["延迟"]["玩家卡住"]
                ):
                    key_up_many()
                    key_down(config["刷图模式"]["快捷键"]["普攻"])
                    time.sleep(0.5)
                    key_up_many()
                    door_list = box_map.get(config["label"]["men"])
                    run = not door_list
                    if s > 10:
                        move_to_target(
                            "卡住了",
                            player_point,
                            rect_center(gs.ss_rect),
                            pad=random.randint(5, 10),
                            run=run
                        )
                    else:
                        random_move(f"卡住了{s:.2f}", run=run, seconds=0.5)
                return player

        gs.player_point_time = time.time()
        gs.player_point_before = gs.player_point
        gs.player_point = player_point
        return player


# endregion


# region main
def predict_source():
    """获取游戏屏幕，然后获取预测结果"""
    global game_img, game_img2

    params = config["程序变量"]["predict"]
    result = yolo_model.predict(
        source=game_img,
        save=False,
        conf=params["conf"],
        device=params["device"],
    )[0]
    # print(result.names)
    boxes = result.boxes.data
    _, box_map = formatted_boxes(boxes, result.names, box_param["分类"])
    # [x1, y1, x2, y2, 概率, 分类坐标]
    # [1413.51, 880.17, 1516.58, 1162.73, 0.88, 1.0],
    # pprint.pp(box_map)

    if config["调试窗口"]["标题"]:
        game_img2 = np.array(
            result.plot(
                labels=params["labels"],
                boxes=params["boxes"],
                line_width=params["line_width"],
                font_size=params["font_size"],
            )
        )

        # gamestart_select_player(draw=True)
        match_img_click(
            area_temp_gray,
            config["程序变量"]["置信度"]["地区传送"],
            click=False,
            draw=True,
        )
        match_img_click(
            select_role_temp_gray,
            config["程序变量"]["置信度"]["选择角色"],
            click=False,
            draw=True,
        )
        match_img_click(
            fb_temp_gray,
            config["程序变量"]["置信度"]["选择副本"],
            click=False,
            draw=True,
        )
        match_img_click(
            close_button_temp_gray,
            0.8,
            click=False,
            draw=True,
        )

        player_list = box_map.get(config["label"]["wanjia"])
        player = None

        if player_list:
            player = sorted(player_list, key=lambda x: x[box_param["概率"]])[-1]

        if player:
            p_point = get_rect_point(player)
            cv2_cross_line(p_point)

            敌人列表 = box_map.get(config["label"]["diren"])
            材料列表 = box_map.get(config["label"]["cailiao"])
            门列表 = box_map.get(config["label"]["men"])
            if 敌人列表:
                for e in 敌人列表:
                    cv2_draw_auxiliary_line(p_point, get_rect_point(e))
            if 材料列表:
                for e in 材料列表:
                    cv2_draw_auxiliary_line(p_point, get_rect_point(e))
            if 门列表:
                for e in 门列表:
                    cv2_draw_auxiliary_line(p_point, get_rect_point(e))

        cv2.rectangle(game_img2, gs.ss_rect[:2], gs.ss_rect[2:], COLORS["green"], 2)
    return box_map


def auto_game(box_map: dict):
    """处理yolo输出"""
    player = find_player(box_map)

    if gs.场景 == "游戏开始":
        gamestart_select_player()
        time.sleep(3)
        gs.场景 = "赛利亚"
        return False
    elif gs.场景 == "赛利亚":
        print("向传送门移动")
        key_down("right")
        time.sleep(3)
        key_up_many()
        gs.场景 = "地区传送"
        return False
    elif gs.场景 == "地区传送":
        if match_img_click(area_temp_gray, config["程序变量"]["置信度"]["地区传送"]):
            print("找到 地区")
            time.sleep(2)

            print("向副本门移动")
            key_down("right")
            time.sleep(5)
            key_up_many()
            gs.场景 = "选择副本"
        else:
            print("没找到 地区")
            key_press("up")
            time.sleep(0.5)

        return False
    elif gs.场景 == "选择副本":
        # 找到副本在哪里
        if match_img_click(fb_temp_gray, config["程序变量"]["置信度"]["选择副本"]):
            print("选择副本，重置难度等级")
            key_press_many("左" * 6)

            print("选择副本等级")
            for _ in range(config["刷图模式"]["选择副本"]["难度"]):
                key_press("right")

            print("进入副本")
            key_press(config["刷图模式"]["快捷键"]["确认"])
            time.sleep(5)
            gs.场景 = "副本中"
        else:
            print("没找到 副本地图")
            key_press("up")
            time.sleep(0.5)

        return False
    elif gs.场景 == "副本中":
        if gs.challenge_again and not box_map.get(config["label"]["fubenditu"]):
            # 刷完图，副本地图会消失
            print("没有疲劳了，返回城镇")
            key_press(config["刷图模式"]["快捷键"]["返回城镇"])

            next_use_player_i = config["刷图模式"]["使用角色"] + 1

            if next_use_player_i >= len(config["角色列表"]):
                print("角色全部刷完，退出程序")
                if config["刷图模式"]["刷完关机"] > 0:
                    s = int(config["刷图模式"]["刷完关机"])
                    print(f"{s}秒后关机")
                    os.system(f"shutdown /s /t {s}")
                return True

            gs.场景 = "切换角色"
            time.sleep(10)
            key_press(config["刷图模式"]["快捷键"]["esc"])
            return False

        enemy_list = box_map.get(config["label"]["diren"])
        materials_list = box_map.get(config["label"]["cailiao"])
        door_list = box_map.get(config["label"]["men"])
        fubenditu_list = box_map.get(config["label"]["fubenditu"])

        if box_map.get(config["label"]["jaingli"]):
            # 领取奖励，捡材料，修装备，再次挑战，一气呵成
            key_up_many()
            快捷键 = config["刷图模式"]["快捷键"]
            延迟 = config["刷图模式"]["延迟"]

            print("领取奖励")
            time.sleep(random.uniform(0.5, 1.0))
            key_press("1")
            time.sleep(random.uniform(0.3, 0.5))
            key_press(快捷键["esc"])
            time.sleep(延迟["领取奖励后"])

            if 快捷键["修理武器"]:
                print("修理武器")
                key_press(快捷键["修理武器"])
                time.sleep(0.5)
                key_press(快捷键["确认"])
                time.sleep(0.5)

            print("移动物品")
            key_press(快捷键["移动物品"])
            time.sleep(延迟["移动物品后"])

            print("确认拾取物品")
            key_down(快捷键["普攻"])
            time.sleep(1)
            key_up_many()
            time.sleep(0.5)

            print("再次挑战")
            key_press(快捷键["再次挑战"])
            time.sleep(延迟["再次挑战后"])

            # 初始化一些状态
            gs.reset_fb_status()
            return

        if not fubenditu_list and gs.player_in_boss_room:
            # print("已通关")
            return False

        if gs.next_room_is_boss() and not gs.player_in_boss_room:
            match_smap_room_loc(fubenditu_list)
            find_to_boss_room_path()
            find_player_current_room_and_next_room()

        if player:
            if enemy_list and not door_list:  # 有门就不打怪了
                key_up_many()
                gs.challenge_again = False
                gs.default_move_room = False
                gs.next_room_time = 0

                if (
                    not gs.player_in_boss_room
                    and gs.next_room_is_boss()
                    and match_img_click(
                        boss_feature_temp_gray,
                        config["程序变量"]["置信度"]["领主特征"],
                        click=False,
                    )
                ):
                    join_boss_room("匹配判断")

                target = find_nearest_target(player, enemy_list, gs.释放距离)
                if target:

                    player_point = get_rect_point(player)
                    target_point = get_rect_point(target)
                    distance = calculate_distance(player_point, target_point)
                    params = {"pad": 50, "run": distance > gs.gw_info["w"] / 3}

                    ok = False
                    # if distance > gs.释放距离:
                    if gs.释放距离 > 0:
                        # 站在目标的x轴上放技能
                        tx, ty = target_point
                        if target_point[0] > player_point[0]:
                            target_point = (tx - gs.释放距离, ty)
                        else:
                            target_point = (tx + gs.释放距离, ty)

                    player_point = move_to_target_y(
                        "敌人y", player_point, target_point, run=params["run"]
                    )
                    player_point = move_to_target_x(
                        "敌人x", player_point, target_point, run=params["run"]
                    )
                    # ok = move_to_target(
                    #     player_point,
                    #     target_point,
                    #     params["pad"],
                    #     run=params["run"],
                    # )
                    player_attack(player_point, target)
                    if config["刷图模式"]["延迟"]["攻击后"] > 0:
                        key_down(config["刷图模式"]["快捷键"]["普攻"])
                        time.sleep(config["刷图模式"]["延迟"]["攻击后"])
                        key_up_many()

                    return False
                else:
                    print("没找到最近的敌人")

            if materials_list:
                key_up_many()
                gs.challenge_again = False
                gs.next_room_time = 0

                target = find_nearest_target(player, materials_list, gs.释放距离)
                if target:
                    player_point = get_rect_point(player)
                    target_point = get_rect_point(target)

                    distance = calculate_distance(player_point, target_point)
                    params = {"run": False, "pad": 0}

                    if distance < random.randint(80, 120):
                        # 距离足够小，斜着走也能捡到
                        move_to_target(
                            "材料1",
                            player_point,
                            target_point,
                            params["pad"],
                            run=params["run"],
                        )
                    else:
                        # 移动y
                        tp = patch_point_y(player_point, target_point)
                        # 判断是否有门
                        y_in_door = line_in_door(player_point, tp, door_list)
                        if not y_in_door:
                            move_to_target(
                                "材料y", player_point, tp, 1, run=params["run"]
                            )
                            player_point = tp
                        else:
                            print(
                                f"路线y与门相交 {'下' if target_point[1] > player_point[1] else '上'}"
                            )

                        # 移动x
                        tp = patch_point_x(player_point, target_point)
                        x_in_door = line_in_door(player_point, tp, door_list)
                        if not x_in_door:
                            move_to_target(
                                "材料x", player_point, tp, 1, run=params["run"]
                            )
                            player_point = tp
                        else:
                            print(
                                f"路线x与门相交 {'右' if target_point[0] > player_point[0] else '左'}"
                            )

                        # 两边都有门
                        if y_in_door and x_in_door:
                            move_to_target(
                                "材料2",
                                player_point,
                                target_point,
                                params["pad"],
                                run=params["run"],
                            )

                    # key_up_many()
                    # key_press(config["刷图模式"]["快捷键"]["普攻"])
                    return False
                else:
                    print("没找到最近的材料")

            if door_list:
                key_up_many()
                gs.challenge_again = False

                # 获取小地图切片
                match_smap_room_loc(fubenditu_list)
                # 从定位分析出前进路线
                find_to_boss_room_path()
                # 当前玩家在路线的哪个位置，以及下个路线的方向
                find_player_current_room_and_next_room()

                handle_door_list(box_map, player)
            else:
                if fubenditu_list and not gs.next_room_is_boss():
                    match_smap_room_loc(fubenditu_list)
                    find_to_boss_room_path()
                    find_player_current_room_and_next_room()

                # 有下一个方向的信息
                nrf = gs.next_room_info()
                if nrf:
                    next_room_pos_zh = nrf[0]
                    if next_room_pos_zh in gs.player_on_screen_pos_zh():
                        # print(f"到了 {next_room_pos_zh}，还是什么都没有")
                        rpos = pos_reverse(next_room_pos_zh)
                        key_down_many(rpos, run=True)
                        print(f"反方向移动:{rpos}")
                        if "up" in rpos or "down" in rpos:
                            time.sleep(random.uniform(3.5, 4.5))
                        else:
                            time.sleep(random.uniform(1.5, 2.5))
                        return
                    else:
                        # 向方向移动
                        move_to_target(
                            "门的方向",
                            gs.player_point,
                            nrf[1],
                            run=True,
                            pad=random.randint(5, 10),
                        )
                        return False

                random_move("只有玩家")

        else:
            # 没找到玩家，但是有之前的point，则移动到ssr中间
            # if gs.player_point_before:
            #     move_to_target(
            #         "ssr中间",
            #         gs.player_point_before,
            #         rect_center(gs.ss_rect),
            #         random.randint(10, 20),
            #         run=False,
            #     )
            #     gs.player_point_before = None
            # else:

            # 可能被敌人遮挡了，释放非boss技能试试
            if enemy_list:
                skill_bar.release_attack(
                    count=1, boss=False, right=bool(random.getrandbits(1))
                )

            random_move("没有玩家")

    elif gs.场景 == "切换角色":
        print("切换角色")

        # 找到选择角色，然后按下去
        if match_img_click(
            select_role_temp_gray, config["程序变量"]["置信度"]["选择角色"]
        ):
            print("找到 选择角色")
            config["刷图模式"]["使用角色"] += 1
            gs.场景 = "游戏开始"
            change_player()
        else:
            print("没找到选择角色按钮")
            key_press(config["刷图模式"]["快捷键"]["esc"])
            time.sleep(1)


def img_to_labelme_file(img, box_obj: dict, index: int, out_dir):
    out_file_name = f"{index}"
    out_img_ext_name = "jpg"
    out_img_filename = f"{out_file_name}.{out_img_ext_name}"
    if not cv2.imwrite(os.path.join(out_dir, out_img_filename), img):
        print("写入img文件失败")
        return False

    height, width = img.shape[:2]

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


def on_press_listener(key):
    from pynput.keyboard import Key

    try:
        if key == gs.swich_key:
            return False

        if key == Key.end:
            gs.pause = not gs.pause
            print("暂停" if gs.pause else "继续")
            return True

        if (
            key == Key.insert
            and os.path.exists(config["截图"]["目录"])
            and game_img is not None
        ):
            op = os.path.join(config["截图"]["目录"], f"{config["截图"]["i"]}.jpg")
            if cv2.imwrite(op, game_img):
                print(f"截图: {op}")
                config["截图"]["i"] += 1
            return True

        step = 10
        shortcut_keys = config["刷图模式"]["快捷键"]
        if key.char == shortcut_keys["加x移速"]:
            gs.move_speed += step
            print(f"加移速:{gs.move_speed}")
        elif key.char == shortcut_keys["减x移速"]:
            gs.move_speed -= step
            print(f"减移速:{gs.move_speed}")

    except AttributeError:
        pass


def predict_game():
    global game_img, game_img2

    title = config["调试窗口"]["标题"]
    if title:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        wh = config["调试窗口"]["宽高"]
        if wh[0] > 0 and wh[1] > 0:
            cv2.resizeWindow(title, wh[0], wh[1])

    with Listener(on_press=on_press_listener) as hk:
        mark_mode_i = config["标记模式"]["begin"]
        mode = config.get("模式")
        if mode == "标记模式":
            out_dir = config["标记模式"]["输出目录"]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=False)

        while hk.running:
            if gs.pause:
                time.sleep(1)
                continue
            game_img = window_capture(
                gs.hwnd, toCv2=True, usePIL=config["程序变量"]["usePIL"]
            )

            # 裁剪掉多余的图像，主要在修炼馆测试
            if mode == "测试移速" and config["测试移速"]["裁剪"]:
                h, w = game_img.shape[:2]
                game_img = game_img[int(h / 2) : h, int(w * 0.2) : w]

            box_map = predict_source()

            # 显示一个测试窗口
            if title:
                cv2.imshow(title, game_img2)
                if config["调试窗口"]["置顶"]:
                    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)

            if mode == "标记模式":
                img_to_labelme_file(game_img, box_map, mark_mode_i, out_dir)
                mark_mode_i += 1
                time.sleep(config["标记模式"]["seconds"])  # 避免生成太多文件

            if mode == "刷图模式":
                if auto_game(box_map):
                    break
                if config["刷图模式"]["延迟"]["截图"] > 0:
                    time.sleep(config["刷图模式"]["延迟"]["截图"])

            if mode == "测试移速":
                player = find_player(box_map)
                if player:
                    testPlayerMoveSpeed.test(player)

    cv2.destroyAllWindows()


def bootstrap():
    global yolo_model, config, gs, testPlayerMoveSpeed
    opts, args = getopt.getopt(
        sys.argv[1:], "ho:r:s:t:m:", ["help", "output=", "conf="]
    )
    print(opts, args)

    if len(args) == 0:
        print("缺少配置文件")
        return

    config = load_json5(args[0])

    for o, a in opts:
        if o == "-r":
            config["刷图模式"]["使用角色"] = int(a)
        if o == "-s":
            config["刷图模式"]["场景"] = a
        if o == "-m":
            config["模式"] = "" if a == "0" else a
        if o == "-t":
            config["调试窗口"]["标题"] = "" if a == "0" else a
        if o == "--conf":
            config["程序变量"]["predict"]["conf"] = float(a)

    # 初始化技能表
    if type(config["技能表"]) is str:
        config["技能表"] = load_json5(config["技能表"])

    gs = GameStatus()

    # pprint.pp(config)
    load_image_temps()
    testPlayerMoveSpeed = TestPlayerMoveSpeed()

    mode = config.get("模式")
    yolo_model = YOLO(os.path.join(config["资源目录"], "best.pt"))
    if mode == "刷图模式":
        change_player()

    with Listener(on_press=lambda key: not (key == gs.swich_key)) as hk:
        while hk.running:
            print(f"选中游戏窗口，按{gs.swich_key}启动")
            time.sleep(1)

    gs.hwnd = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(gs.hwnd)
    if window_title.find(config["程序变量"]["游戏窗口标题"]) != -1:
        predict_game()
    key_up_many()


# endregion

if __name__ == "__main__":
    try:
        bootstrap()
    except KeyboardInterrupt:
        pass
