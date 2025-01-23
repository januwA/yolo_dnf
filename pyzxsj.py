import datetime
import os
from typing import Mapping

from demo_km_macro import reverse_macro_all

# 取消yolo打印信息
os.environ["YOLO_VERBOSE"] = str(False)

import sys
from PySide6 import QtCore, QtWidgets, QtGui
import string
import time
from PIL import Image, ImageGrab
import cv2
import numpy as np
import pyautogui
from pynput.keyboard import Listener, Key
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

pause = False


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


def match_img(img, temp, conf=0.9, to_center=False, ret_list=False):
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
自动释放技能 = False
yolo_model = None
game_hwnd = None


def on_press_listener(key):
    global pause, save_img_i, 自动释放技能
    if key == Key.delete:
        return False

    if key == Key.end:
        pause = not pause
        print("暂停" if pause else "继续")
        return True

    if key == Key.page_down:
        自动释放技能 = not 自动释放技能
        return True

    # 游戏截图
    if key == Key.insert:
        cv2.imwrite(
            os.path.join(
                r"C:\Users\16418\Desktop\zxsj\game_imgs",
                f"{save_img_random_string}_{save_img_i}.jpg",
            ),
            window_capture(game_hwnd, False),
        )
        save_img_i += 1
        return True


def 释放技能(_lst: list[str] = None, one=False):
    """只按快捷键, one循环完一次就退出"""
    lst = (
        ["1", "2", "3", "4", "5", "6", "q", "e", "r", "t", "f1", "f2", "f3"]
        if _lst is None or len(_lst) == 0
        else _lst
    )
    i = 0
    pydirectinput.press(快捷键["自动选择目标"])
    while True:
        if pause:
            break

        pydirectinput.press(lst[i])
        i += 1
        if i >= len(lst):
            if one:
                break
            i = 0
            pydirectinput.press(快捷键["自动选择目标"])
        time.sleep(random.uniform(0.1, 0.3))


def 领取战令():
    exe_hotkey(快捷键["战令"])
    pyautogui.click(626, 901, duration=random.uniform(1, 2))  # 领取按钮
    pyautogui.click(958, 612, duration=random.uniform(1, 2))  # 确定按钮
    time.sleep(1)
    exe_hotkey(快捷键["esc"])


def 赠送好友礼物():
    exe_hotkey(快捷键["仙友"])
    pyautogui.click(545, 428, duration=random.uniform(1, 2))  # 聊天

    loc = None
    h = 80  # 每个高度
    b_h = 366
    while True:
        loc = 检测_聊天玫瑰花()
        # 如果没有玫瑰花，则可能是系统消息
        if loc is None:
            pyautogui.click(708, b_h, duration=random.uniform(1, 2))  # 点击第二个聊天
            b_h += h
        else:
            break

    for _ in range(3):
        pyautogui.click(loc[0], loc[1], duration=random.uniform(1, 2))  # 赠送玫瑰花

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


def 检测_已通关():
    """从图片中检测，最小化窗口按钮，返回按钮的中心点"""
    img = window_capture(game_hwnd)
    img_h, img_w = img.shape[:2]
    loc = match_img(
        img,
        imread2(
            r"C:\Users\16418\Desktop\zxsj\config\通关_最小化窗口.jpg",
            "通关_最小化窗口",
        ),
        conf=0.5,
        to_center=True,
    )
    if loc is not None and loc[0] > img_w / 2 and loc[1] > img_h / 2:
        return loc
    return None


def 检测_聊天玫瑰花():
    img = window_capture(game_hwnd)
    loc = match_img(
        img,
        imread2(
            r"C:\Users\16418\Desktop\zxsj\config\聊天_玫瑰花.jpg",
            "聊天_玫瑰花",
        ),
        conf=0.7,
        to_center=True,
    )
    return loc


def 检测_治疗救援():
    """副本中阵亡，会显示【治疗救援】图片"""
    img = window_capture(game_hwnd)
    loc = match_img(
        img,
        imread2(
            r"C:\Users\16418\Desktop\zxsj\config\治疗救援.jpg",
            "治疗救援",
        ),
        conf=0.7,
        to_center=True,
    )
    return loc


def 检测_原地疗伤():
    """阵亡，会显示【原地疗伤】图片"""
    img = window_capture(game_hwnd)
    loc = match_img(
        img,
        imread2(
            r"C:\Users\16418\Desktop\zxsj\config\原地疗伤.jpg",
            "原地疗伤",
        ),
        conf=0.7,
        to_center=True,
    )
    return loc


def 检测_就近疗伤():
    """阵亡，会显示【就近疗伤】图片"""
    img = window_capture(game_hwnd)
    loc = match_img(
        img,
        imread2(
            r"C:\Users\16418\Desktop\zxsj\config\就近疗伤.jpg",
            "就近疗伤",
        ),
        conf=0.7,
        to_center=True,
    )
    return loc


def 检查_技能距离不够():
    """技能距离不够，会显示灰色"""
    img = window_capture(game_hwnd)
    img_h, img_w = img.shape[:2]
    loc = match_img(
        img,
        imread2(
            r"C:\Users\16418\Desktop\zxsj\config\距离不够_技能灰色.jpg",
            "距离不够_技能灰色",
        ),
        to_center=True,
    )
    if loc is not None and loc[1] > img_h / 2:
        return loc
    return None


def 检查_副本退出按钮():
    img = window_capture(game_hwnd)
    img_h, img_w = img.shape[:2]
    loc = match_img(
        img,
        imread2(
            r"C:\Users\16418\Desktop\zxsj\config\副本中_退出按钮.jpg",
            "副本中_退出按钮",
        ),
        to_center=True,
    )
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
        # reverse_macro_all(
        #     r"C:\Users\16418\Desktop\zxsj\config\开启焚香谷_副本.json",
        #     random.randint(1, 3),
        # )
        exe_hotkey(快捷键["副本"])
        time.sleep(random.uniform(1, 2))
        pyautogui.click(1109, 217, duration=random.uniform(1, 2))  # 5人小队
        pyautogui.click(1207, 145, duration=random.uniform(1, 2))  # 普通

        match_玄火烬八荒 = match_img(
            window_capture(game_hwnd),
            imread2(
                r"C:\Users\16418\Desktop\zxsj\config\玄火烬八荒.jpg",
                "玄火烬八荒",
            ),
            conf=0.7,
        )
        if match_玄火烬八荒 is None:
            pyautogui.click(1347, 199, duration=random.uniform(1, 2))  # 隐藏大型副本
            pyautogui.click(1484, 495, duration=random.uniform(1, 2))  # 智能队友
        else:
            # 直接找到智能队友,点击
            rectes = match_img(
                window_capture(game_hwnd),
                imread2(
                    r"C:\Users\16418\Desktop\zxsj\config\智能队友.jpg",
                    "智能队友",
                ),
                conf=0.7,
                ret_list=True,
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

        print("进入副本后向前移动")
        pydirectinput.keyDown(快捷键["前"])
        time.sleep(5)
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

        point = 检测_已通关()
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
        rect = match_img(
            img,
            imread2(
                r"C:\Users\16418\Desktop\zxsj\config\打坐图标.jpg",
                "打坐图标",
            ),
            conf=0.8,
        )
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
    point = match_img(
        window_capture(),
        imread2(r"C:\Users\16418\Desktop\zxsj\config\乐器_琴.jpg", "乐器_琴"),
        0.7,
        to_center=True,
    )
    if point is not None:
        pyautogui.click(point[0], point[1], duration=1, button=pyautogui.RIGHT)
        time.sleep(5)

    pyautogui.click(922, 1015, duration=random.uniform(1, 2))  # 打开乐谱
    pyautogui.click(483, 219, duration=random.uniform(1, 2))  # 试听按钮
    time.sleep(10)
    exe_hotkey(快捷键["esc"])


def 演奏_笛():
    exe_hotkey(快捷键["包裹"])
    point = match_img(
        window_capture(),
        imread2(r"C:\Users\16418\Desktop\zxsj\config\乐器_笛.jpg", "乐器_笛"),
        0.7,
        to_center=True,
    )
    if point is not None:
        pyautogui.click(point[0], point[1], duration=1, button=pyautogui.RIGHT)
        time.sleep(5)

    pyautogui.click(922, 1015, duration=random.uniform(1, 2))  # 打开乐谱
    pyautogui.click(483, 219, duration=random.uniform(1, 2))  # 试听按钮
    time.sleep(10)
    exe_hotkey(快捷键["esc"])


def 挂机按f():
    while True:
        if pause:
            return False
        pydirectinput.press("f")
        time.sleep(random.uniform(0.5, 1))  # 进度条


def 猜拳():
    while True:
        if pause:
            return False
        pyautogui.click(1346, 665, duration=random.uniform(0.3, 0.5))  # 剪刀
        time.sleep(random.uniform(0.5, 1))


def 钓鱼():
    # 检查是否持竿
    if (
        match_img(
            window_capture(),
            imread2(r"C:\Users\16418\Desktop\zxsj\config\鱼_持竿.jpg", "鱼_持竿"),
            0.7,
            to_center=True,
        )
        is None
    ):
        # 打开背包，找到鱼竿，然后右键
        exe_hotkey(快捷键["包裹"])
        time.sleep(random.uniform(1, 2))
        point = match_img(
            window_capture(),
            imread2(r"C:\Users\16418\Desktop\zxsj\config\青竹鱼竿.jpg", "青竹鱼竿"),
            0.7,
            to_center=True,
        )
        if point is None:
            print("没找到鱼竿")
            return False
        else:
            print("持竿")
            pyautogui.rightClick(point[0], point[1], duration=random.uniform(1, 2))
            time.sleep(random.uniform(1, 2))

    print("抛竿")
    pydirectinput.keyDown("space")
    time.sleep(random.uniform(0.6, 1))
    pydirectinput.keyUp("space")

    time.sleep(random.uniform(2, 3))

    print("等待鱼")

    # 等待鱼，一直按空格直到鱼上钩
    while True:
        if pause:
            return False
        if (
            match_img(
                window_capture(),
                imread2(r"C:\Users\16418\Desktop\zxsj\config\鱼_上钩.jpg", "鱼_上钩"),
                0.7,
                to_center=True,
            )
            is not None
        ):
            print("鱼上钩")
            break

        pydirectinput.press("space")  # 收杆
        time.sleep(random.uniform(0.3, 0.5))

    print("开始收杆")

    while True:
        if pause:
            return False

        if (
            match_img(
                window_capture(),
                imread2(r"C:\Users\16418\Desktop\zxsj\config\鱼_收杆.jpg", "鱼_收杆"),
                0.7,
            )
            is None
        ):
            pydirectinput.keyUp("space")
            print("收杆结束")
            break

        # if (
        #     match_img(
        #         window_capture(),
        #         imread2(r"C:\Users\16418\Desktop\zxsj\config\鱼_挣扎.jpg", "鱼_挣扎"),
        #         0.7,
        #     )
        #     is not None
        # ):
        #     print("鱼挣扎")
        #     pydirectinput.keyUp("space")
        #     time.sleep(random.uniform(0.5, 1))
        #     pydirectinput.keyDown("space")
        #     time.sleep(random.uniform(1.8, 2))

        pydirectinput.keyDown("space")
        time.sleep(random.uniform(1.5, 2))
        pydirectinput.keyUp("space")
        time.sleep(random.uniform(0.3, 0.5))


def bootstrap():
    global pause, yolo_model, game_hwnd

    with Listener(on_press=lambda key: not (key == Key.delete)) as hk:
        while hk.running:
            print("按delete开始")
            time.sleep(1)

    # yolo_model = YOLO(r"C:\Users\16418\Desktop\zxsj\矿石\train\weights\best.pt")
    game_hwnd = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(game_hwnd)
    print(window_title)  # ZhuxianClient

    with Listener(on_press=on_press_listener) as hk:
        while hk.running:
            if pause:
                time.sleep(1)
                continue

            焚香谷_副本()

            # if 自动释放技能:
            #     释放技能(["1", "2", "3"])

            # game_img = window_capture(game_hwnd)
            # result = yolo_model.predict(
            #     source=game_img,
            #     save=False,
            #     conf=0.7,
            #     device="0",
            # )[0]
            # boxes = result.boxes.data
            # _, box_map = formatted_boxes(boxes, result.names)
            # game_img = np.array(result.plot())
            # cv2.imshow("game_img", game_img)
            # cv2.waitKey(1)

            # time.sleep(random.uniform(0.1, 0.3))

    cv2.destroyAllWindows()


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
        self.绑定游戏窗口.clicked.connect(self.get_game_hwdn)
        self.停止按钮.clicked.connect(self.stop_event)
        self.功能选择.addItems(
            [
                "焚香谷副本",
                "自动释放技能",
                "领取战令",
                "赠送好友礼物",
                "修改个性签名",
                "演奏 琴",
                "演奏 笛",
                "挂机按f",
                "北荒战云_前往天衡传送门",
                "钓鱼",
                "点击鼠标左键",
                "点击鼠标右键",
                "猜拳",
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

        # 绑定快捷键
        shortcut_end = QtGui.QShortcut(QtGui.QKeySequence("end"), self)
        shortcut_end.activated.connect(self.handle_shortcut_end)

    @QtCore.Slot()
    def handle_shortcut_end(self):
        if pause:
            self.start()
        else:
            self.stop_event()

    @QtCore.Slot()
    def stop_event(self):
        global pause
        pause = True

    @QtCore.Slot()
    def get_game_hwdn(self):
        global game_hwnd

        time.sleep(3)
        game_hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(game_hwnd)
        print(window_title)
        self.绑定游戏窗口.setText(window_title)
        rect = win32gui.GetWindowRect(game_hwnd)
        left, top, right, bottom = rect
        w = right - left
        h = bottom - top
        if w != 1920 and h != 1080:
            msg_box = QtWidgets.QMessageBox()
            msg_box.setText("游戏分辨率必须为1920x1080全屏模式")
            msg_box.exec()

    @QtCore.Slot()
    def start(self):
        global game_hwnd, pause

        param1 = self.param1.text()

        if not game_hwnd:
            print("没找到游戏窗口")
            return

        action = self.功能选择.currentText()
        win32gui.SetForegroundWindow(game_hwnd)
        time.sleep(1)
        pause = False
        if action == "焚香谷副本":

            def f1():
                while True:
                    if pause:
                        break
                    焚香谷_副本()
                print("焚香谷_副本 结束")

            threading.Thread(target=f1).start()
        elif action == "自动释放技能":
            threading.Thread(target=释放技能).start()
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
        elif action == "挂机按f":
            threading.Thread(target=挂机按f).start()
        elif action == "北荒战云_前往天衡传送门":
            reverse_macro_all(
                r"C:\Users\16418\Desktop\zxsj\config\北荒战云_前往天衡传送门.json", 1
            )
        elif action == "点击鼠标左键":
            time.sleep(3)
            for _ in range(int(param1)):
                pydirectinput.click()
                time.sleep(0.01)

        elif action == "点击鼠标右键":
            time.sleep(3)
            for _ in range(int(param1)):
                pydirectinput.click(button=pydirectinput.RIGHT)
                time.sleep(0.01)
        elif action == "猜拳":
            threading.Thread(target=猜拳).start()


def on_about_to_quit():
    global pause
    print("应用程序即将关闭")
    pause = True


# 游戏 1920x1080 全屏
# 技能释放方式：在目标位置释放
# 移动方向: 镜头面向
# 技能释放方向: 角色面向
# 悬停施法：关闭
if __name__ == "__main__":
    time.sleep(2)
    # loc = 检测_聊天玫瑰花()
    # if loc:
    #     pyautogui.moveTo(loc[0], loc[1])
    # else:
    #     pyautogui.moveTo(2, 2)

    # img = window_capture()
    # rectes = match_img(
    #     img,
    #     imread2(
    #         r"C:\Users\16418\Desktop\zxsj\config\智能队友.jpg",
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

    app = QtWidgets.QApplication([])
    app.aboutToQuit.connect(on_about_to_quit)
    widget = MyWidget()
    # widget.resize(800, 300)
    widget.show()
    sys.exit(app.exec())
