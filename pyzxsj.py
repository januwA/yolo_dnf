import time
from pynput.keyboard import Listener, Key
import pydirectinput
import random

pause = False

快捷键 = {
    "打坐": ["alt", "f"],
    "玄机瞳": ["alt", "e"],
    "御兽": ["alt", "w"],
    "御剑": ["alt", "q"],
    "esc": 'esc',
    "地图": 'm',
}


def exe_hotkey(hk):
    if type(hk) == list:
        pydirectinput.keyDown(hk[0])
        time.sleep(random.uniform(0.05, 0.15))
        for i in range(1, len(hk)):
            pydirectinput.press(hk[i])
            time.sleep(random.uniform(0.05, 0.15))
        pydirectinput.keyUp(hk[0])


def on_press_listener(key):
    global pause
    if key == Key.delete:
        return False

    if key == Key.end:
        pause = not pause
        print("暂停" if pause else "继续")
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


def bootstrap():
    global pause
    with Listener(on_press=lambda key: not (key == Key.delete)) as hk:
        while hk.running:
            print("按delete开始")
            time.sleep(1)

    with Listener(on_press=on_press_listener) as hk:
        while hk.running:
            if pause:
                time.sleep(1)
                continue

            释放技能()

            # time.sleep(random.uniform(0.1, 0.3))


if __name__ == "__main__":
    try:
        bootstrap()
        # exe_hotkey(快捷键["打坐"])
    except KeyboardInterrupt:
        pass
