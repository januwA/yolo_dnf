import datetime
import os

# 取消yolo打印信息
os.environ["YOLO_VERBOSE"] = str(False)


import string
import time
from PIL import Image, ImageGrab
import cv2
import numpy as np
import pyautogui
from pynput.keyboard import Listener, Key
import pydirectinput
import random
from ultralytics import YOLO
import win32gui
import win32ui
import win32con
import win32api
import win32print


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


def move_and_click(pos: tuple[int], sleep_ab=(0.5, 1)):
    """移动鼠标到pos然后点击"""
    pyautogui.moveTo(pos[0], pos[1])  # 更多选项
    time.sleep(random.uniform(sleep_ab[0], sleep_ab[1]))
    pyautogui.click()
    time.sleep(random.uniform(sleep_ab[0], sleep_ab[1]))


def window_capture(hwnd, usePIL=False):
    hwndDC = win32gui.GetWindowDC(hwnd)

    # 获取窗口尺寸
    rect = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = rect
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


pause = False
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


技能_list = [
    {
        "key": "1",
        "施法": 1.5,
        "冷却": 0,
    },
    {
        "key": "2",
        "施法": 1.5,
        "冷却": 9.5,
    },
    {
        "key": "3",
        "施法": 0.5,
        "冷却": 15,
    },
    {
        "key": "4",
        "施法": 2,
        "冷却": 5.7,
    },
    {
        "key": "5",
        "施法": 0.7,
        "冷却": 0,
    },
    {
        "key": "e",
        "施法": 1.3,
        "冷却": 20,
    },
    {
        "key": "r",
        "施法": 0.5,
        "冷却": 7.6,
    },
    {
        "key": "f2",
        "施法": 6,
        "冷却": 115,
    },
    {
        "key": "f3",
        "施法": 0.5,
        "冷却": 60,
    },
]
技能_i = 0


def 释放技能():
    global 技能_list, 技能_i
    技能 = 技能_list[技能_i]
    now = time.time()
    if now - 技能.get("释放时间", 0) > 技能.get("冷却", 0):
        pydirectinput.press(技能["key"])
        技能["释放时间"] = now
        time.sleep(技能.get("施法", 0))  # 等待施法
    技能_i += 1
    if 技能_i >= len(技能_list):
        技能_i = 0


def 领取战令():
    exe_hotkey(快捷键["战令"])
    time.sleep(random.uniform(1, 2))
    move_and_click((840, 1470))
    exe_hotkey(快捷键["esc"])


def 赠送好友礼物():
    exe_hotkey(快捷键["仙友"])
    time.sleep(random.uniform(1, 2))

    move_and_click((826, 716))  # 聊天

    for i in range(random.randint(3, 5)):
        move_and_click((1546, 1194))  # 赠送玫瑰花

    # 退出面板
    exe_hotkey(快捷键["esc"])


def 修改个性签名():
    exe_hotkey(快捷键["个人信息"])
    time.sleep(random.uniform(1, 2))

    move_and_click((134, 784))  # 更多选项
    move_and_click((150, 1034))  # 修改个人签名
    move_and_click((1586, 848))  # 个人签名

    # 删除已有签名
    for i in range(random.randint(20, 25)):
        pyautogui.press("backspace")
    time.sleep(random.uniform(1, 2))
    for i in range(random.randint(20, 25)):
        pyautogui.press("delete")
    time.sleep(random.uniform(1, 2))

    today = datetime.date.today()
    pyautogui.typewrite(str(today), interval=random.uniform(0.3, 0.5))
    time.sleep(random.uniform(0.5, 1))

    move_and_click((1346, 950))  # 确定

    # 退出面板
    exe_hotkey(快捷键["esc"])


def bootstrap():
    global pause, yolo_model, game_hwnd

    with Listener(on_press=lambda key: not (key == Key.delete)) as hk:
        while hk.running:
            print("按delete开始")
            time.sleep(1)

    yolo_model = YOLO(r"C:\Users\16418\Desktop\zxsj\矿石\train\weights\best.pt")
    game_hwnd = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(game_hwnd)
    print(window_title)  # ZhuxianClient

    with Listener(on_press=on_press_listener) as hk:
        while hk.running:
            if pause:
                time.sleep(1)
                continue

            if 自动释放技能:
                释放技能()

            game_img = window_capture(game_hwnd, True)
            result = yolo_model.predict(
                source=game_img,
                save=False,
                conf=0.7,
                device="0",
            )[0]
            boxes = result.boxes.data
            _, box_map = formatted_boxes(boxes, result.names)
            game_img = np.array(result.plot())
            cv2.imshow("game_img", game_img)
            cv2.waitKey(1)

            # time.sleep(random.uniform(0.1, 0.3))

    cv2.destroyAllWindows()


# 游戏性，战斗及操作，移动方向，镜头面向
if __name__ == "__main__":
    try:
        bootstrap()

        # time.sleep(3)
        # pydirectinput.keyDown(快捷键['上旋转镜头'])
        # 修改个性签名()
        # 赠送好友礼物()
        # 领取战令()
    except KeyboardInterrupt:
        pass
