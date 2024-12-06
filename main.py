import getopt
import json
import os
import sys

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
from PIL import Image, ImageGrab

from sklearn.cluster import KMeans

# import pyautogui 不要用这个库，cv2.imshow比例显示会有问题

# class #############################################################################################################


# 技能
class Skill:
    def __init__(self, key: str, cooling_time, aoe=False, mk=None):
        self.key = key
        self.cooling_time = cooling_time
        self.aoe = aoe
        self.release_t = 0
        self.mk = mk
        if type(self.mk) == int:
            self.mk = [self.key for _ in range(self.mk)]

    def can_release(self):
        return time.time() - self.release_t > self.cooling_time

    def release(self):
        if not self.can_release():
            return False

        key_press(self.key)
        if self.mk:
            time.sleep(0.1)
            for k in self.mk:
                key_press(k)
                time.sleep(0.1)
        self.release_t = time.time()
        return True


# 测试玩家移动速度，确保在大地图中测试
class TestPlayerMoveSpeed:
    def __init__(self):
        self.from_point = None
        self.to_point = None
        self.move_seconds = 1

    def test(self, 玩家, keys, run):
        p_point = get_rect_point(玩家)
        if not self.from_point:
            self.from_point = p_point
            key_down_many(keys, zhKey=True, run=run)
            time.sleep(self.move_seconds)
            key_up_many(keys, zhKey=True)
        elif self.from_point and not self.to_point:
            self.to_point = p_point
            distance = calculate_distance(self.from_point, self.to_point)
            移速 = distance / self.move_seconds
            print(f"移速:{移速}")
            return 移速


class GameStatus:
    def __init__(self):
        # 一些固定的属性
        self._boss_bgr = (21, 20, 139)
        self._boss_tolerance = 20

        self._player_bgr = (208, 94, 51)
        self._player_tolerance = 30

        # 一些初始化的属性，在每次reset将初始化这些属性
        self._smap_grid: list[list[int]] = None
        self._to_boss_path_list: list[tuple[int, int]] = None
        self._room_road_block: dict = {(0, 0): [(0, 1)]}

        self.move_speed = {"x": 240, "y": 170}
        self.winname = ""
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

        # 游戏窗口内的矩形信息 (x1, y1, x2, y2)，这个矩形用来定位玩家在屏幕上的那个位置
        self.ss_rect = (0, 0, 0, 0)

        # 0可通行，1不可通行，TODO：处理随机生成的地图
        self.smap_grid = self._smap_grid
        # self.map_grid = [
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        # ]
        self.smap_item_size = 36
        self.smap_item_size_scal = 0
        self.smap_main_len = 0
        self.smap_cross_len = 0

        self.smap_points = None
        self.smap_items_rect = None
        self.room_i = 0
        self.to_boss_path_list: list | None = self._to_boss_path_list
        self.room_road_block: dict[tuple, list[tuple]] = self._room_road_block
        self.next_room_info: tuple[str, tuple[int, int]] = None  # 下一个房间的位置信息

        self.boss_room_point_list = []  # 有可能有多个boss图标
        self.boss_room_point = None  # 最后boss
        self.player_room_point = None  # 玩家在小地图上的位置
        self.player_room_point_before = None  # 上一个房间的位置

        self.player_not_find = False
        self.player_point = None
        self.player_point_before = None
        self.player_point_time = 0
        self.player_in_boos_room = False
        self.player_on_screen_pos_zh = ""
        self.场景 = ""  # 副本中
        self.疲劳值 = 100
        self.after_seconds = 0  # 过去了多少秒

    def reset_fb_status(self):
        # 如果是固定地图，则不用重置
        self.smap_points = None
        self.to_boss_path_list = self._to_boss_path_list

        # 这些都应该是必须初始化的
        self.room_i = 0
        self.场景 = ""
        self.boss_room_point_list.clear()
        self.boss_room_point = None
        self.player_room_point = None
        self.player_point = None
        self.player_point_before = None
        self.player_point_time = 0
        self.player_on_screen_pos_zh = ""
        self.player_in_boos_room = False
        self.next_room_info = None
        self.room_road_block = self._room_road_block
        self.after_time = 0
        self.player_in_boos_room = False

        self.smap_grid = self._smap_grid

    def reset_path_status(self):
        self.to_boss_path_list = self._to_boss_path_list
        self.player_room_point = None
        self.next_room_info = None
        self.room_i = 0


# global #############################################################################################################

testPlayerMoveSpeed = TestPlayerMoveSpeed()
gs = GameStatus()

yolo_model = None

boos_temp = cv2.imdecode(
    np.fromfile(r"./image_template/boss_icon.png", dtype=np.uint8),
    -1,
)
boos_temp_gray = cv2.cvtColor(boos_temp, cv2.COLOR_BGR2GRAY)

player_temp = cv2.imdecode(
    np.fromfile(r"./image_template/player_icon.png", dtype=np.uint8),
    -1,
)
player_temp_gray = cv2.cvtColor(player_temp, cv2.COLOR_BGR2GRAY)


VK = {"f11": 0x7A}

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

user32 = ctypes.windll.user32

skill_bar = [
    # Skill("a", 1, mk=4),
    Skill("s", 8, aoe=True),
    Skill("d", 32, aoe=True),
    Skill("f", 26, aoe=True),
    Skill("g", 19, aoe=True),
    Skill("h", 24, aoe=True),
    Skill("q", 5, mk=["z"]),
    Skill("q", 5, mk=["c", "x"]),
    Skill("q", 5, mk=["x"]),
    Skill("w", 26, aoe=True),
    Skill("e", 11, aoe=True),
    Skill("r", 19, aoe=True),
]
skill_bar_buff = ["alt", "ctrl"]

random_moves_list = ["左上", "左下", "右上", "右下", "上", "右", "下", "左"]

# 记录所有按下的key
key_down_list = []

# tools #############################################################################################################


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


# 小地图房间坐标方向
def get_smap_move_zh_pos(current_node, next_node):
    m1, c1 = current_node
    m2, c2 = next_node

    t_point = None
    x1, y1, x2, y2 = gs.ss_rect

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    pad = 50

    if m2 > m1:
        t_point = (xc, y2 + pad)
        return "下", t_point
    elif m2 < m1:
        t_point = (xc, y1 - pad)
        return "上", t_point
    elif c2 > c1:
        t_point = (x2 + pad, yc)
        return "右", t_point
    else:
        t_point = (x1 - pad, yc)
        return "左", t_point


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


def get_ssr_pos_zh(point):
    x, y = point
    pos_zh = "中"
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


def formatted_boxes(boxes, names):
    boxes2 = []
    box_obj = {}
    for box in boxes:
        fb = [float(f"{nun:.2f}") for nun in box.tolist()]
        ti = names[int(fb[box_param["分类"]])]
        if ti not in box_obj or box_obj[ti] is None:
            box_obj[ti] = []
        box_obj[ti].append(fb)
        boxes2.append(fb)
    return boxes2, box_obj


# 获取矩形的某个点
def get_rect_point(rect, centerRate=None):
    if not centerRate:
        centerRate = 0.95

    (x1, _, x2, y2) = rect[:4]
    x = (x1 + x2) / 2
    # center_y = (y1 + y2) / 2
    y = y2 * centerRate
    return int(x), int(y)


def rect_size(rect):
    (x1, y1, x2, y2, _, _) = rect
    return int(x2 - x1), int(y2 - y1)


# 计算两点之间的距离
def calculate_distance(p_point, t_point):
    (x1, y1) = p_point
    (x2, y2) = t_point

    dx = x1 - x2
    dy = y1 - y2

    distance = math.sqrt(dx**2 + dy**2)

    return distance


# 计算两点之间的距离和角度
def calculate_distance_and_angle(p_point, t_point):
    (x1, y1) = p_point
    (x2, y2) = t_point

    math.atanh

    dx = x1 - x2
    dy = y1 - y2

    distance = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)  # 使用atan2函数可以更准确地计算角度，包括象限
    return distance, math.degrees(angle)


# 寻找最近的target
def find_nearest_target(player, targets_list):
    min_distance = float("inf")
    p_point = get_rect_point(player)

    nearest = None

    for target in targets_list:
        t_point = get_rect_point(target)

        distance = calculate_distance(p_point, t_point)
        if distance < min_distance:
            min_distance = distance
            nearest = target

    return nearest


# 计算矩形的中心点坐标
def rect_center(rect):
    (x1, y1, x2, y2, _, _) = rect
    return (x1 + x2) / 2, (y1 + y2) / 2


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

    gs.ss_rect = (int(w * 0.2), int(h * 0.55), int(w * 0.75), int(h * 0.75))

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


def key_press(key):
    # print(f"点击: {key}")
    pydirectinput.press(key)


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
        key_press(keys[0])
        time.sleep(0.1)
        key_down_many(keys)
    else:
        for k in keys:
            key_down(k)
            time.sleep(0.1)


def random_move(logname, run=False):
    print(f"随机游走 {logname}")
    zh_keys = random.choice(random_moves_list)
    key_down_many(zh_keys, zhKey=True, run=run)
    time.sleep(0.3)
    key_up_many()


def player_attack(player, target, x=False, aoe=False):
    # 使技能前，抬起所有键，避免卡住
    key_up_many()

    if player and target:
        # 向哪个方向攻击
        px, py = gs.player_point
        tx, ty = get_rect_point(target)

        if tx > px:
            key_press("right")
        else:
            key_press("left")
        pass

    if x:
        key_down("x")
        time.sleep(2)
        key_up("x")
    else:
        ok = False
        for skill in skill_bar:
            ok = skill.release()
            if ok:
                break

        # 没有释放任何技能
        if not ok:
            key_down("x")
            time.sleep(1)
            key_up("x")


def init_smap_items_rect_grid(small_map, pad: int):
    if gs.player_in_boos_room:
        return

    if gs.player_room_point and gs.boss_room_point:
        return

    if small_map:
        small_map = sorted(small_map, key=lambda x: x[box_param["概率"]])[-1]

    if not small_map:
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
    if not gs.smap_items_rect or gs.player_in_boos_room:
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
                boos_res = cv2.matchTemplate(
                    smap_item_img_gray, boos_temp_gray, cv2.TM_CCOEFF_NORMED
                )
                boos_loc = np.where(boos_res >= 0.8)
                if len(list(zip(*boos_loc[::-1]))):
                    boss_locs.append(
                        (
                            smap_point,
                            np.max(boos_res),
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

    if len(player_locs):
        gs.player_room_point = sorted(player_locs, key=lambda x: x[1])[-1][0]
    else:
        gs.player_room_point = None

    if not gs.boss_room_point and len(boss_locs):
        gs.boss_room_point_list = boss_locs
        gs.boss_room_point = sorted(boss_locs, key=lambda x: x[1])[-1][0]

    print(gs.player_room_point, gs.boss_room_point)


def get_move_key(p_point, t_point, onle=None, pad=0):
    px, py = p_point
    tx, ty = t_point
    k = None
    if onle == "x":
        if tx > px:
            k = "right"
        elif tx < px:
            k = "left"
    elif onle == "y":
        if ty > py:
            k = "down"
        elif ty < py:
            k = "up"
    else:
        if tx > px:
            k = "right"
        elif tx < px:
            k = "left"
        elif ty > py:
            k = "down"
        elif ty < py:
            k = "up"
    return k


# 将角度转换为方向
def degrees2PosZh(degrees, en=False, 容差=15):
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

    if (0 - 容差) < degrees < (0 + 容差):
        keys = "L" if en else "左"
    elif (90 - 容差) < degrees < (90 + 容差):
        keys = "T" if en else "上"
    elif ((180 - 容差) < degrees <= 180) or (-180 < degrees <= (-180 + 容差)):
        keys = "R" if en else "右"
    elif (-90 - 容差) < degrees < (-90 + 容差):
        keys = "B" if en else "下"

    return keys


# 检查目标是否在ssr矩形外
def is_out_ssr(next_room_pos_zh: str, target, isPoint=False):
    dx, dy = target if isPoint else get_rect_point(target)
    (x1, y1, x2, y2) = gs.ss_rect

    # 检测门是否在正确的边缘外
    if next_room_pos_zh == "上" and dy < y1:
        return True
    if next_room_pos_zh == "右" and dx > x2:
        return True
    if next_room_pos_zh == "下" and dy > y2:
        return False
    if next_room_pos_zh == "左" and dx < x1:
        return False

    return False


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

    return door_point

    # if not door:
    #     return None

    # if is_out_ssr(next_room_pos_zh, door):
    #     return door
    # else:
    #     return None


def find_player_current_room_and_next_room():
    if gs.player_in_boos_room:
        return

    if gs.player_room_point and gs.to_boss_path_list:
        pre_room_i = gs.room_i

        try:
            room_i = gs.to_boss_path_list.index(gs.player_room_point)

            # 房间变化了
            # if room_i < pre_room_i or room_i > pre_room_i:
            #     print(
            #         f"进入{ '下' if room_i > pre_room_i else '上' }个房间, {pre_room_i}到{room_i} {gs.player_room_point}"
            #     )

            # 到达下个房间先加buff
            if room_i > pre_room_i:
                key_press_many(skill_bar_buff)

            gs.room_i = room_i

            # 获取下个房间位置
            next_room = gs.to_boss_path_list[room_i + 1]
            gs.next_room_info = get_smap_move_zh_pos(
                gs.player_room_point,
                next_room,
            )
        except:
            key_up_many()
            print("走错房间, 未在规定路线内")
            gs.reset_path_status()
            return
    else:
        # 到了boss房间是找不到玩家的位置的
        if (
            not gs.player_room_point
            and gs.boss_room_point
            and gs.to_boss_path_list
            and gs.room_i == len(gs.to_boss_path_list) - 2
        ):
            if not gs.player_in_boos_room:
                print("BOOS 房间")
                gs.player_in_boos_room = True
                # TODO:优化释放终极技能
                key_press("y")


def find_to_boss_room_path():
    # 寻找最佳路线
    if gs.player_room_point and gs.boss_room_point and not gs.to_boss_path_list:
        gs.to_boss_path_list = a_star_search(
            gs.player_room_point,
            gs.boss_room_point,
            gs.smap_grid,
            gs.room_road_block,
        )
        if gs.to_boss_path_list:
            print(
                f"{gs.smap_main_len}x{gs.smap_cross_len} 路线: {gs.to_boss_path_list} 玩家:{gs.player_room_point} BOSS:{gs.boss_room_point}"
            )


# 处理门列表
def handle_door_list(box_map: dict, player):
    if gs.player_in_boos_room:
        return

    door_list = box_map.get("门")
    if not door_list:
        return

    smap = box_map.get("副本地图")
    if not smap:
        return

    if not gs.player_room_point or not gs.boss_room_point:
        # 尽可能快的找到玩家在小地图上的位置
        for i in range(3):
            init_smap_items_rect_grid(smap, random.randint(-i, +1))
            if gs.player_room_point and gs.boss_room_point:
                break
            time.sleep(0.3)

    # 有下一个方向的信息
    if gs.next_room_info:
        next_room_pos_zh = gs.next_room_info[0]
        print(f"下个房间方向: {next_room_pos_zh}")

        # 走到位了
        if next_room_pos_zh in gs.player_on_screen_pos_zh:
            gs.after_seconds = 0
            key_up_many()

            target_point = find_door(next_room_pos_zh, door_list)
            if target_point:
                if move_to_target(
                    player, target_point, pad=50, run=False, target_is_point=True
                ):
                    key_up_many()
            else:
                pass
                print("没找到们")
                # if gs.to_boss_path_list and gs.player_room_point:
                #     key_up_many()
                #     print(
                #         f"没找到门，重新定义路线，下个房间:{gs.to_boss_path_list[gs.room_i + 1]} {gs.next_room_info}"
                #     )
                #     (mi, ci) = gs.to_boss_path_list[gs.room_i + 1]
                #     gs.smap_grid[mi][ci] = 1
                #     gs.reset_path_status()
        else:
            # 说明这条路不通的
            if gs.after_seconds != 0 and time.time() - gs.after_seconds > 10:
                ok = False
                for door in door_list:
                    ok = is_out_ssr(next_room_pos_zh, door)
                    if ok:
                        break
                print(f"还没有到{next_room_pos_zh}，匹配门: {ok}")
                if not ok:
                    if gs.player_room_point not in gs.room_road_block:
                        gs.room_road_block[gs.player_room_point] = []

                    next_room_point = gs.to_boss_path_list[gs.room_i + 1]
                    gs.room_road_block[gs.player_room_point].append(next_room_point)
                    gs.after_seconds = 0
                    gs.to_boss_path_list = None  # 重置路径
                    gs.room_i = 0
                    return

            else:
                gs.after_seconds = time.time()
            screen_point = gs.next_room_info[1]
            move_to_target(
                gs.player_point,
                screen_point,
                run=True,
                pad=10,
                player_is_point=True,
                target_is_point=True,
            )


# 移动到目标返回true
def move_to_target(
    player,
    target,
    pad=None,
    run=False,
    player_is_point=False,
    target_is_point=False,
    centerRate=None,
    slow=False,
):
    if not pad:
        pad = random.uniform(10.0, 200.0)

    p_point = player if player_is_point else get_rect_point(player, centerRate)
    t_point = target if target_is_point else get_rect_point(target, centerRate)

    distance, degrees = calculate_distance_and_angle(p_point, t_point)

    if distance <= pad:
        # print("太近不移动")
        return True

    # 太近了就不跑了
    if distance < 200:
        run = False

    keys = degrees2PosZh(degrees)

    if keys:
        speed_key = "y" if keys == "上" or keys == "下" else "x"

        move_speed = gs.move_speed[speed_key]
        if slow:
            move_speed /= 2

        s_t = distance / move_speed

        if run:
            s_t /= 2

        key_down_many(keys, zhKey=True, run=run)
        time.sleep(s_t)
        key_up_many()

    return False


def find_player(player_list):
    if player_list:
        gs.player_not_find = False
        # 获取概率最大的玩家
        player = sorted(player_list, key=lambda x: x[box_param["概率"]])[-1]

        # 在屏幕上的点位
        player_point = get_rect_point(player)

        if gs.player_point and player_point:
            # 检查两次的位置是否一样，如果被建筑卡住这很常见
            distance = calculate_distance(player_point, gs.player_point)
            if (
                distance <= 5  # 距离变化太低
                and gs.player_point_time != 0  # 之前有设置
                and time.time() - gs.player_point_time > 5  # 过去了5秒
            ):
                key_up_many()
                random_move("可能卡住了")

        gs.player_point_before = gs.player_point
        gs.player_point = player_point

        gs.player_point_time = time.time()

        # 处于屏幕哪个位置 上中下左？
        gs.player_on_screen_pos_zh = get_ssr_pos_zh(gs.player_point)

        return player
    else:
        gs.player_not_find = True


# main #############################################################################################################


def predict_image():
    # 加载图片
    im1 = Image.open(
        "C:\\Users\\16418\\Desktop\\shenDianWaiWei\\1_segment\\YOLODataset\\images\\val\\00000003.jpg"
    )
    data = {
        "enemy": [
            [144.21, 452.0, 222.8, 544.11, 0.57, 1.0],
            [629.55, 454.65, 705.54, 531.92, 0.94, 1.0],
        ],
        "player": [[299.78, 334.76, 402.29, 450.09, 0.88, 0.0]],
    }

    t = find_nearest_target(data["player"][0], data["enemy"])
    print(t)
    print(data)

    # result = model.predict(source=im1, save=True, conf=0.25, device=0)[0]
    # pprint.pprint(formatted_boxes(result.boxes.data, result.names))
    # orig_shape: (598, 1067)


# 获取游戏屏幕，然后获取预测结果
def predict_source(source):
    result = yolo_model.predict(
        source=source,
        save=False,
        conf=0.25,
        device=0,
    )[0]
    # print(result.names)
    img = np.array(result.plot())
    boxes = result.boxes.data
    # [2.2865e-01, 7.8152e+02, 2.2020e+02, 1.1498e+03, 9.8804e-01, 2.0000e+00],
    # pprint.pp(boxes)

    _, box_map = formatted_boxes(boxes, result.names)

    # [x1, y1, x2, y2, 概率, 分类坐标]
    # [1413.51, 880.17, 1516.58, 1162.73, 0.88, 1.0],
    # pprint.pp(boxes2)

    if gs.winname:
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


# 处理yolo输出
def auto_game(box_map: dict, img):
    player = find_player(box_map.get("玩家"))
    enemy_list = box_map.get("敌人")
    materials_list = box_map.get("材料")
    door_list = box_map.get("门")

    if box_map.get("选择界面"):
        gs.场景 = "选择界面"
    elif box_map.get("赛利亚"):
        gs.场景 = "赛利亚"
    elif box_map.get("返回城镇"):
        gs.场景 = "选择副本"
    # 尽可能放在后面判断，应为城镇也有门
    else:
        gs.场景 = "副本中"

    # print(f'场景:{GAME_STATUS.场景}')

    if gs.场景 == "副本中":
        if box_map.get("奖励"):
            key_up_many()

            print("领取奖励")
            time.sleep(1)
            key_press("esc")

            # 等待3秒
            time.sleep(3)

            print("移动物品")
            key_press("0")  # 移动物品快捷键
            time.sleep(2)

            # 初始化一些状态
            gs.reset_fb_status()

            # 继续挑战
            key_press("f10")
            print("按下f10")

            # 等会避免多余的动作
            time.sleep(5)

            return

        # 获取小地图切片
        init_smap_items_rect_grid(box_map.get("副本地图"), 0)
        # 从小地图切片中找到定位
        smap_find_player_and_boss_point(img)
        # 从定位分析出前进路线
        find_to_boss_room_path()
        # 当前玩家在路线的哪个位置，以及下个路线的方向
        find_player_current_room_and_next_room()

        if player:
            if enemy_list and not door_list:  # 有门就不打怪了
                target = find_nearest_target(player, enemy_list)
                if target:
                    move_to_target(player, target, pad=50, run=True)
                    player_attack(player, target)
                    time.sleep(0.5)

            elif materials_list and not gs.player_in_boos_room:  # boss房间追后拾取材料
                target = find_nearest_target(player, materials_list)
                if target:
                    move_to_target(player, target, pad=10, run=False)
                    key_up_many()
                    key_press("x")

            elif door_list and not gs.player_in_boos_room:  # 在boss房间不管门
                handle_door_list(box_map, player)
            else:
                random_move("除了玩家，其他什么都没有")

        else:
            # 没找到玩家，但是有之前的point，则移动到ssr中间
            if gs.player_point_before:
                move_to_target(
                    gs.player_point_before,
                    gs.ss_rect,
                    pad=10,
                    run=True,
                    player_is_point=True,
                )
            else:
                random_move("没有玩家")


def img_to_labelme_file(img, box_obj: dict, index: int, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    out_file_name = f"{index}"
    out_img_ext_name = "jpg"
    out_img_filename = f"{out_file_name}.{out_img_ext_name}"
    if not cv2.imwrite(os.path.join(out_dir, out_img_filename), img):
        print("写入img文件失败")
        return False

    height, width, channels = img.shape

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

    # cv2.namedWindow(gs.winname, cv2.WINDOW_NORMAL)

    with pynput.keyboard.Listener(on_press=lambda key: not (key == gs.swich_key)) as hk:
        loop_i = 1
        while hk.running:
            # 截取的游戏屏幕图片
            in_img = None
            try:
                in_img = window_capture(gs.hwnd, toCv2=True)
            except:
                in_img = window_capture(gs.hwnd, toCv2=True, usePIL=True)

            # 游戏屏幕图片交给模型处理，得到输出的数据，输出的图片带有学习标记
            (img, box_obj) = predict_source(in_img)

            img_to_labelme_file(
                in_img, box_obj, loop_i, r"C:\Users\16418\Desktop\风暴幽城\segment3"
            )
            loop_i += 1
            time.sleep(1)  # 避免生成太多文件

            # 处理这些数据来自动打怪
            # if auto_game(box_obj, in_img):
            #     break

            # 显示一个测试窗口
            if gs.winname:
                cv2.imshow(gs.winname, img)
                # cv2.resizeWindow(gs.winname, 700, 400)
                cv2.setWindowProperty(gs.winname, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)
    cv2.destroyAllWindows()


def bootstrap():
    global yolo_model

    opts, args = getopt.getopt(sys.argv[1:], "ho:", ["help", "output="])

    if len(args) >= 1:
        gs.winname = args[0]

    # for opt, arg in opts:
    #     if opt in ("-h", "--help"):
    #         print("帮助信息")
    #         sys.exit()
    #     elif opt in ("-o", "--output"):
    #         output_file = arg
    #         print("输出文件：", output_file)

    yolo_model = YOLO(r"C:\Users\16418\Desktop\风暴幽城\trains\train\weights\best.pt")

    with pynput.keyboard.Listener(on_press=lambda key: not (key == gs.swich_key)) as hk:
        while hk.running:
            print(f"选中游戏窗口，按{gs.swich_key}启动")
            time.sleep(1)

    gs.hwnd = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(gs.hwnd)
    if window_title.find("地下城与勇士") != -1:
        predict_game()

    key_up_many()


if __name__ == "__main__":
    bootstrap()
