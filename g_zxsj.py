import datetime
import json
import os
from typing import Mapping

from demo_km_macro import reverse_macro_all

# 取消yolo打印信息
os.environ["YOLO_VERBOSE"] = str(False)

import sys
from PySide6 import QtCore, QtWidgets
import string
import time
from PIL import Image, ImageGrab
import cv2
import numpy as np
import pyautogui
from pynput.keyboard import Key, GlobalHotKeys
import pydirectinput

pydirectinput.FAILSAFE = False
import random

# from ultralytics import YOLO
import win32gui
import win32ui
import win32con
import win32api
import win32print
import threading
from plyer import notification

pause = True

keydown_list = []


def key_up(key):
    if key not in keydown_list:
        return
    keydown_list.remove(key)
    pydirectinput.keyUp(key)


def key_down(key):
    global keydown_list
    if key in keydown_list:
        return

    keydown_list.append(key)
    pydirectinput.keyDown(key)


def 修改系统分辨率(width=1920, height=1080):
    """
    设置屏幕分辨率。

    Args:
        width: 宽度像素值。
        height: 高度像素值。
    """
    # 获取当前显示器设置
    dm = win32api.EnumDisplaySettings(None, 0)

    # 设置新的分辨率
    dm.PelsWidth = width
    dm.PelsHeight = height

    # 应用更改
    win32api.ChangeDisplaySettings(dm, 0)
    # win32api.ChangeDisplaySettingsEx(dm, 0, win32con.CDS_UPDATEREGISTRY)


def rect_center(rect):
    """计算矩形的中心点坐标"""
    (x1, y1, x2, y2) = rect[:4]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# 读取缓存，有缓存就直接返回
read2map: Mapping[str, cv2.typing.MatLike] = {}


def imread2(p: str, cache_key: str) -> cv2.typing.MatLike:
    global read2map
    if cache_key in read2map:
        return read2map[cache_key]

    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), -1)
    read2map[cache_key] = img
    return img


def match_img(img, temp, conf=0.9, to_center=True, ret_list=False, splice=None):
    if img is None:
        img = window_capture()

    if splice is not None and len(splice) == 4:
        h, w = img.shape[:2]
        img = img[
            int(h * splice[0]) : int(h * splice[1]),
            int(w * splice[2]) : int(w * splice[3]),
        ]

    if type(temp) is str:
        temp = imread2(rf"C:\zxsj\config\{temp}.jpg", temp)

    temp_h, temp_w = temp.shape[:2]
    res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
    # print(f"{maxVal=}")
    loc = np.where(res >= conf)
    loc_lst = list(zip(*loc[::-1]))
    if len(loc_lst):
        if ret_list:
            rect_lst = list(
                map(
                    lambda loc: (loc[0], loc[1], loc[0] + temp_w, loc[1] + temp_h),
                    loc_lst,
                )
            )
            if to_center:
                return list(map(lambda rect: rect_center(rect), rect_lst))
            return rect_lst

        if to_center:
            maxLoc = rect_center(
                (
                    maxLoc[0],
                    maxLoc[1],
                    maxLoc[0] + temp_w,
                    maxLoc[1] + temp_h,
                )
            )
        return maxLoc
    return None


def formatted_boxes(boxes, names, type_index: int = 5):
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


def window_capture(hwnd=None, usePIL=True, game_rect=None):
    if hwnd is None:
        hwnd = win32gui.GetForegroundWindow()

    hwndDC = win32gui.GetWindowDC(hwnd)

    # 获取窗口尺寸
    rect = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = rect
    if game_rect is not None:
        game_rect["left"] = left
        game_rect["top"] = top
        game_rect["right"] = right
        game_rect["left"] = left
        game_rect["bottom"] = bottom

    w = right - left
    h = bottom - top

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
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


快捷键 = {
    "esc": "esc",
    # 角色控制
    "前": "w",
    "后": "s",
    "左": "a",
    "右": "d",
    "上": "space",
    "下": "v",
    "轻功攻击": "x",
    "左旋转镜头": "left",
    "右旋转镜头": "right",
    "上旋转镜头": "up",
    "下旋转镜头": "down",
    "切换跑步_走路": "/",
    "自动奔跑": "=",
    "御剑": ["alt", "q"],
    "御兽": ["alt", "w"],
    "玄机瞳": ["alt", "e"],
    "打坐": ["alt", "f"],
    "自动选择目标": "tab",
    "队伍快捷标记": ["ctrl", "1"],
    # 界面开关
    "包裹": "b",
    "任务": "l",
    "地图": "m",
    "角色": "c",
    "技能": "k",
    "法宝": "u",
    "星魄": ["alt", "x"],
    "境界": "[",
    "副本": "i",
    "招募平台": ["alt", "i"],
    "阵营": ["alt", "n"],
    "帮派": "n",
    "竞技": ["alt", "j"],
    "仙友": "o",
    "探索": "j",
    "成就": "y",
    "图鉴": ["alt", "t"],
    "排行榜": ["alt", "p"],
    "绘卷": ["alt", "h"],
    "宠物": "p",
    "星历": ["alt", "v"],
    "个人信息": ["alt", "g"],
    "风华值": ["alt", "l"],
    "商店": ";",
    "商城": ["alt", "m"],
    "仙府": ["alt", "o"],
    "生产手册": "\\",
    "战令": "-",
    "休闲动作": ["alt", "b"],
}


def exe_hotkey(hk):
    if type(hk) == str and "+" in hk and len(hk) > 1:
        hk = hk.split("+")

    if type(hk) == list:
        pydirectinput.keyDown(hk[0])
        time.sleep(random.uniform(0.05, 0.15))
        for i in range(1, len(hk)):
            pydirectinput.press(hk[i])
            time.sleep(random.uniform(0.05, 0.15))
        pydirectinput.keyUp(hk[0])
    else:
        pydirectinput.press(hk)


save_img_i = 0
save_img_random_string = "".join(random.choice(string.ascii_letters) for _ in range(6))
game_hwnd = None


def 释放技能(_lst: list[str] = None, one=False, 旋转镜头=False):
    """只按快捷键, one循环完一次就退出"""
    lst = (
        [
            快捷键["自动选择目标"],
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            # "q", "e", 保命技能
            "r",
            "t",
            # "f1",
            # "f2",
            # "f3",
        ]
        if _lst is None or len(_lst) == 0
        else _lst
    )
    i = 0

    if 旋转镜头:
        pydirectinput.keyDown(快捷键["右旋转镜头"])

    lst_len = len(lst)
    while True:
        if pause:
            break

        if lst[i].startswith("m:"):  # 点击鼠标
            pydirectinput.click(
                button=pydirectinput.LEFT if lst[i] == "m:l" else pydirectinput.RIGHT
            )
        else:  # 点击键盘
            exe_hotkey(lst[i])
        i += 1
        if i >= lst_len:
            if one:
                break
            i = 0

        if lst_len > 1:
            time.sleep(random.uniform(0.05, 0.1))

    if 旋转镜头:
        pydirectinput.keyUp(快捷键["右旋转镜头"])


def 领取战令():
    exe_hotkey(快捷键["战令"])
    pyautogui.click(626, 901, duration=random.uniform(1, 2))  # 领取按钮
    pyautogui.click(958, 612, duration=random.uniform(1, 2))  # 确定按钮
    time.sleep(1)
    exe_hotkey(快捷键["esc"])


def 赠送好友礼物():
    exe_hotkey(快捷键["仙友"])
    pyautogui.click(545, 428, duration=random.uniform(1, 2))  # 聊天

    point = None
    h = 80  # 每个高度
    b_h = 366
    while True:
        point = match_img(None, "聊天_玫瑰花", conf=0.7)
        # 如果没有玫瑰花，则可能是系统消息
        if point is None:
            pyautogui.click(708, b_h, duration=random.uniform(1, 2))  # 点击第二个聊天
            b_h += h
        else:
            break

    for _ in range(3):
        pyautogui.click(point[0], point[1], duration=random.uniform(1, 2))  # 赠送玫瑰花

    time.sleep(1)
    exe_hotkey(快捷键["esc"])


def 修改个性签名():
    exe_hotkey(快捷键["个人信息"])
    pyautogui.click(82, 467, duration=random.uniform(1, 2))  # 更多选项
    pyautogui.click(104, 619, duration=random.uniform(1, 2))  # 修改个人签名
    pyautogui.click(965, 508, duration=random.uniform(1, 2))  # 个人签名

    # 删除已有签名
    for i in range(random.randint(20, 25)):
        pyautogui.press("backspace")
    for i in range(random.randint(20, 25)):
        pyautogui.press("delete")
    time.sleep(random.uniform(0.3, 1))

    today = datetime.date.today()
    pyautogui.typewrite(str(today), interval=random.uniform(0.3, 0.5))
    time.sleep(random.uniform(0.5, 1))

    pyautogui.click(855, 570, duration=random.uniform(1, 2))  # 确定
    time.sleep(1)
    exe_hotkey(快捷键["esc"])


def 检测_治疗救援():
    """副本中阵亡，会显示【治疗救援】图片"""
    return match_img(None, "治疗救援", conf=0.7)


def 检测_原地疗伤():
    """阵亡，会显示【原地疗伤】图片"""
    return match_img(None, "原地疗伤", conf=0.7)


def 检测_就近疗伤():
    """阵亡，会显示【就近疗伤】图片"""
    return match_img(None, "就近疗伤", conf=0.7)


def 检查_副本退出按钮():
    img = window_capture(game_hwnd)
    img_h, img_w = img.shape[:2]
    loc = match_img(img, "副本中_退出按钮")
    if loc is not None and loc[0] > img_w / 2 and loc[1] < img_h / 2:
        return loc
    return None


def 打坐_回血(seconds: float = 15):
    exe_hotkey(快捷键["打坐"])
    time.sleep(seconds)
    exe_hotkey(快捷键["打坐"])


def 焚香谷_副本():
    def check_in(check_count: int):
        """检查是否在副本中"""
        _check_count = 0
        while True:
            if pause:
                return False

            if check_count > 0 and _check_count > check_count:
                return False

            if 检查_副本退出按钮() is not None:
                break

            _check_count += 1
            time.sleep(random.uniform(1, 3))
        return True

    point = 检测_原地疗伤()
    if point is not None:
        pyautogui.click(point[0], point[1], duration=random.uniform(1, 2))
        time.sleep(3)
        打坐_回血(30)

    if not check_in(1):
        print("准备进入副本")
        exe_hotkey(快捷键["副本"])
        time.sleep(random.uniform(1, 2))
        pyautogui.click(1109, 217, duration=random.uniform(1, 2))  # 5人小队
        pyautogui.click(1207, 145, duration=random.uniform(1, 2))  # 普通

        if match_img(None, "玄火烬八荒", conf=0.7) is None:
            pyautogui.click(1347, 199, duration=random.uniform(1, 2))  # 隐藏大型副本
            pyautogui.click(1484, 495, duration=random.uniform(1, 2))  # 智能队友
        else:
            # 直接找到智能队友,点击
            rectes = match_img(
                None, "智能队友", conf=0.7, to_center=False, ret_list=True
            )
            if type(rectes) is list:
                for rect in rectes:
                    x1, y1, x2, y2 = rect
                    if y1 < 495 < y2:
                        pyautogui.click(
                            x1 + int((x2 - x1) / 2), 495, duration=random.uniform(1, 2)
                        )  # 智能队友
                        break
            else:
                print("没有找到只能队友")
                return False

        pyautogui.click(865, 564, duration=random.uniform(1, 2))  # 确定

        time.sleep(10)  # 等10秒进度条在检查

        # 一直检测直到进入副本
        check_in(-1)
        pydirectinput.keyDown(快捷键["前"])
        time.sleep(6)
        pydirectinput.keyUp(快捷键["前"])

    # 将鼠标移动到左上角
    pyautogui.moveTo(2, 2)

    while True:
        if pause:
            return False

        point = 检测_治疗救援()
        if point is not None:
            pyautogui.click(point[0], point[1], duration=random.uniform(1, 2))
            time.sleep(random.uniform(1, 2))
            pyautogui.moveTo(2, 2)

        point = match_img(None, "通关_最小化窗口", conf=0.5)
        if point is not None:
            print("检测到已通关")
            time.sleep(1)
            pyautogui.click(
                960, 609, duration=random.uniform(1, 2)
            )  # 点击确定，复活次数不够
            pyautogui.click(point[0], point[1], duration=random.uniform(1, 2))
            time.sleep(1)
            break

        释放技能(None, one=True)

        time.sleep(random.uniform(0.2, 0.5))

    time.sleep(1)
    print("通关后退出")
    pyautogui.click(1625, 241, duration=random.uniform(1, 2))  # 离开按钮
    pyautogui.click(865, 564, duration=random.uniform(1, 2))  # 确认按钮
    pyautogui.click(865, 564, duration=random.uniform(1, 2))  # 确认按钮

    print("确认退出进度条结束")
    time.sleep(10)
    while True:
        if pause:
            return False

        img = window_capture(game_hwnd)
        img_h, img_w = img.shape[:2]
        rect = match_img(img, "打坐图标", conf=0.8)
        if rect is not None and rect[1] > img_h / 2:
            break
        time.sleep(random.uniform(0.2, 0.5))

    打坐时间 = 15
    point = 检测_原地疗伤()
    if point is not None:
        pyautogui.click(point[0], point[1], duration=random.uniform(1, 2))
        time.sleep(3)
        打坐时间 = 30

    print("打坐")
    打坐_回血(打坐时间)

    return True


def 演奏_琴():
    exe_hotkey(快捷键["包裹"])
    point = match_img(None, "乐器_琴", 0.7)
    if point is not None:
        pyautogui.click(point[0], point[1], duration=1, button=pyautogui.RIGHT)
        time.sleep(5)

    pyautogui.click(922, 1015, duration=random.uniform(1, 2))  # 打开乐谱
    pyautogui.click(483, 219, duration=random.uniform(1, 2))  # 试听按钮
    time.sleep(10)
    exe_hotkey(快捷键["esc"])


def 演奏_笛():
    exe_hotkey(快捷键["包裹"])
    point = match_img(None, "乐器_笛", 0.7)
    if point is not None:
        pyautogui.click(point[0], point[1], duration=1, button=pyautogui.RIGHT)
        time.sleep(5)

    pyautogui.click(922, 1015, duration=random.uniform(1, 2))  # 打开乐谱
    pyautogui.click(483, 219, duration=random.uniform(1, 2))  # 试听按钮
    time.sleep(10)
    exe_hotkey(快捷键["esc"])


def 点击鼠标(n: int, button: str):
    for _ in range(n):
        if pause:
            break
        pydirectinput.click(button=button)
        time.sleep(0.01)


def 猜拳():
    while True:
        if pause:
            return False
        pyautogui.click(1346, 665, duration=random.uniform(0.3, 0.5))  # 剪刀
        time.sleep(random.uniform(0.5, 1))


def 钓鱼():
    _key = "space"
    pydirectinput.keyUp(_key)

    # 检查是否持竿
    if not match_img(None, "鱼_持竿", 0.7, splice=(0.5, 1, 0, 1)):
        # 打开背包，找到鱼竿，然后右键
        exe_hotkey(快捷键["包裹"])
        time.sleep(random.uniform(1, 2))
        point = match_img(None, "青竹鱼竿", 0.7)
        if point is None:
            print("没找到鱼竿")
            return False
        else:
            print("持竿")
            pydirectinput.rightClick(point[0], point[1], duration=random.uniform(1, 2))
            time.sleep(random.uniform(1, 2))

    print("抛竿")
    key_down(_key)
    time.sleep(random.uniform(0.6, 1))
    key_up(_key)

    time.sleep(random.uniform(2, 3))

    print("等待鱼")

    # 等待鱼，一直按空格直到鱼上钩
    while True:
        if pause:
            return False
        img = window_capture()
        h, w = img.shape[:2]
        img = img[0 : int(h * 0.3), int(w * 0.2) : int(w * 0.8)]

        if match_img(img, "鱼_上钩", 0.7) and match_img(img, "鱼_收杆", 0.7):
            print("鱼上钩")
            break

        pydirectinput.press(_key)  # 收杆
        time.sleep(random.uniform(0.1, 0.2))

    print("开始收杆")

    key_down(_key)
    time.sleep(random.uniform(1, 2))
    key_up(_key)

    while True:
        if pause:
            return False

        img = window_capture()
        h, w = img.shape[:2]
        img = img[0 : int(h * 0.3), 0:w]

        鱼_收杆 = match_img(img, "鱼_收杆", 0.7)
        if not 鱼_收杆:
            key_up(_key)
            print("收杆结束")
            break

        for _ in range(5):
            if pause:
                return False
            key_down(_key)
            time.sleep(random.uniform(0.3, 0.5))
            key_up(_key)
            time.sleep(random.uniform(0.05, 0.1))

        time.sleep(random.uniform(0.1, 0.3))


def 常驻签到():
    exe_hotkey(快捷键["esc"])
    point = match_img(None, "活动菜单", 0.7)
    if point is None:
        return False

    pyautogui.click(point[0], point[1], duration=random.uniform(1, 2))
    time.sleep(random.uniform(1, 2))

    point = match_img(None, "活动常驻", 0.7)
    pyautogui.click(point[0], point[1], duration=random.uniform(1, 2))

    pyautogui.click(447, 319, duration=random.uniform(1, 2))  # 每日签到
    pyautogui.click(1522, 810, duration=random.uniform(1, 2))  # 签到
    time.sleep(random.uniform(0.5, 1))
    exe_hotkey(快捷键["esc"])
    time.sleep(random.uniform(0.5, 1))
    exe_hotkey(快捷键["esc"])


def 剑青云连招():
    global pause
    # 技能 = {
        # "穿星": ["f1", 1.5],
        # "御剑诀": ["f2", 0],
        # "贯虹": ["f3", 0],
        # "青霜剑华": ["1", 0],
        # "怒剑劫": ["2", 0],
        # "镇魔剑罡": ["3", 0],
        # "剑拂云": ["4", 1.2],
    # }
    技能 = {}
    with open(r"C:\zxsj\config\剑青云_物理.json", encoding="utf-8") as f:
        技能 = json.load(f)

    普攻2 = [
        技能["御剑诀"],
        技能["御剑诀"],
    ]
    普攻4 = 普攻2 + 普攻2
    青霜_贯虹 = [
        技能["青霜剑华"],
        技能["贯虹"],
    ] + 普攻4

    # 释放穿星前用普工，用于滑动镜头
    穿星_贯虹 = (
        普攻2
        + [
            技能["穿星"],
            技能["贯虹"],
        ]
        + 普攻2
    )

    # 第1个穿星
    连招 = [技能["贯虹"]] + 普攻4
    连招 += 青霜_贯虹
    连招 += [技能["怒剑劫"], 技能["贯虹"]] + 普攻4
    连招 += 青霜_贯虹
    连招 += [技能["镇魔剑罡"], 技能["贯虹"]]
    连招 += 穿星_贯虹

    # 第2个穿星
    连招 += [技能["剑拂云"], 技能["贯虹"]] + 普攻4
    # 释放 剑拂云 免气用一次青霜
    连招 += 青霜_贯虹
    连招 += 青霜_贯虹
    连招 += 青霜_贯虹
    连招 += 穿星_贯虹

    while True:
        for k in 连招:
            if len(k) != 2:
                continue

            if pause:
                return

            if k[0] == "left":
                pydirectinput.keyDown(k[0])
                time.sleep(k[1])
                pydirectinput.keyUp(k[0])
            else:
                pydirectinput.press(k[0])
                if k[1] > 0:
                    time.sleep(k[1])


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.col_layout = QtWidgets.QVBoxLayout(self)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.启动按钮 = QtWidgets.QPushButton("确定")
        self.停止按钮 = QtWidgets.QPushButton("停止")
        self.绑定游戏窗口 = QtWidgets.QPushButton("绑定游戏窗口")
        self.功能选择 = QtWidgets.QComboBox()

        self.启动按钮.clicked.connect(self.start)
        self.绑定游戏窗口.clicked.connect(lambda: self.get_game_hwdn(3))
        self.停止按钮.clicked.connect(self.stop_event)
        self.功能选择.addItems(
            [
                "剑青云连招",
                "释放技能",
                "释放技能_旋转镜头",
                "焚香谷副本",
                "领取战令",
                "赠送好友礼物",
                "修改个性签名",
                "演奏 琴",
                "演奏 笛",
                "北荒战云_前往天衡传送门",
                "钓鱼",
                "猜拳",
                "常驻签到",
                "修改系统分辨率",
            ]
        )

        self.layout.addWidget(self.绑定游戏窗口)
        self.layout.addWidget(self.功能选择)
        self.layout.addWidget(self.启动按钮)
        self.layout.addWidget(self.停止按钮)
        self.col_layout.addLayout(self.layout)

        # 参数行
        self.layout2 = QtWidgets.QHBoxLayout(self)
        self.param1 = QtWidgets.QLineEdit(self)
        self.layout2.addWidget(self.param1)
        self.col_layout.addLayout(self.layout2)

        h = GlobalHotKeys(
            {
                "<end>": self.handle_shortcut_end,
                # "<f4>": 剑青云连招,
            }
        )
        h.start()

    # @QtCore.Slot()
    def handle_shortcut_end(self):
        action = self.功能选择.currentText()
        if pause:
            notification.notify(
                title="zxsj",
                message=f"开启脚本:{action}",
                app_icon=None,  # 通知图标，可选
                timeout=1,  # 通知显示时长，单位为秒，可选
            )
            if not game_hwnd:
                self.get_game_hwdn(0)
            self.start()
        else:
            notification.notify(
                title="zxsj",
                message=f"停止脚本:{action}",
                app_icon=None,  # 通知图标，可选
                timeout=1,  # 通知显示时长，单位为秒，可选
            )
            self.stop_event()

    @QtCore.Slot()
    def stop_event(self):
        global pause
        pause = True

    @QtCore.Slot()
    def get_game_hwdn(self, _sleep=3):
        global game_hwnd

        if _sleep:
            time.sleep(_sleep)

        game_hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(game_hwnd)
        print(window_title)
        notification.notify(
            title="zxsj",
            message=f"已绑定窗口:{game_hwnd} {window_title}",
            app_icon=None,  # 通知图标，可选
            timeout=1,  # 通知显示时长，单位为秒，可选
        )
        self.绑定游戏窗口.setText(f"{window_title}")
        # rect = win32gui.GetWindowRect(game_hwnd)
        # left, top, right, bottom = rect
        # w = right - left
        # h = bottom - top
        # if w != 1920 and h != 1080:
        #     msg_box = QtWidgets.QMessageBox()
        #     msg_box.setText("游戏分辨率必须为1920x1080全屏模式")
        #     msg_box.exec()

    @QtCore.Slot()
    def start(self):
        global game_hwnd, pause
        param1 = self.param1.text().strip()

        if game_hwnd:
            win32gui.SetForegroundWindow(game_hwnd)

        pause = False

        action = self.功能选择.currentText()
        if action == "焚香谷副本":

            def f1():
                while True:
                    if pause:
                        break
                    焚香谷_副本()
                print("焚香谷_副本 结束")

            threading.Thread(target=f1).start()
        elif action == "释放技能" or action == "释放技能_旋转镜头":
            args = [None, False, False]
            if param1:
                args[0] = param1.split(",")

            if action == "释放技能_旋转镜头":
                args[2] = True

            threading.Thread(target=释放技能, args=args).start()
        elif action == "剑青云连招":
            threading.Thread(target=剑青云连招).start()
        elif action == "领取战令":
            threading.Thread(target=领取战令).start()
        elif action == "赠送好友礼物":
            threading.Thread(target=赠送好友礼物).start()
        elif action == "修改个性签名":
            threading.Thread(target=修改个性签名).start()
        elif action == "钓鱼":

            def f1():
                while True:
                    if pause:
                        break
                    钓鱼()
                print("钓鱼 结束")

            threading.Thread(target=f1).start()
        elif action == "演奏 琴":
            threading.Thread(target=演奏_琴).start()
        elif action == "演奏 笛":
            threading.Thread(target=演奏_笛).start()
        elif action == "北荒战云_前往天衡传送门":
            reverse_macro_all(r"C:\zxsj\config\北荒战云_前往天衡传送门.json", 1)
        elif action == "猜拳":
            threading.Thread(target=猜拳).start()
        elif action == "常驻签到":
            threading.Thread(target=常驻签到).start()
        elif action == "修改系统分辨率":
            修改系统分辨率()


def on_about_to_quit():
    global pause
    print("应用程序即将关闭")
    pause = True


def bootstrap():
    app = QtWidgets.QApplication([])
    app.aboutToQuit.connect(on_about_to_quit)
    widget = MyWidget()
    # widget.resize(800, 300)
    widget.show()
    sys.exit(app.exec())


# 游戏 1920x1080 全屏
# 技能释放方式：在目标位置释放
# 传统模式，移动方向: 镜头面向
# 传统模式，技能释放方向: 角色面向
# 悬停施法：关闭
# pyinstaller -F --hidden-import plyer.platforms.win.notification g_zxsj.py
if __name__ == "__main__":
    # loc = 检测_聊天玫瑰花()
    # if loc:
    #     pyautogui.moveTo(loc[0], loc[1])
    # else:
    #     pyautogui.moveTo(2, 2)

    # img = window_capture()
    # rectes = match_img(
    #     img,
    #     imread2(
    #         r"C:\zxsj\config\智能队友.jpg",
    #         "智能队友",
    #     ),
    #     conf=0.7,
    #     ret_list=True,
    # )
    # for rect in rectes:
    #     x1, y1, x2, y2 = rect
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    bootstrap()
