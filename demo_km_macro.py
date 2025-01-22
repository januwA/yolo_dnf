import json
import re
import time
import winsound
import pyautogui
import pynput
from pynput.keyboard import Key, Listener, KeyCode
from pynput.mouse import Button, Controller
import pydirectinput
import random

# 初始化变量
exit_app_key = Key.delete
is_recording = False
actions = []
outfile = "actions.json"


def key_to_string(key):
    if isinstance(key, Key):
        Key.alt_gr
        k = str(key).replace("Key.", "")  # Remove "Key." prefix
        pattern = r"_."
        return re.sub(pattern, "", k)  # 不区分左右
    elif isinstance(key, KeyCode):
        return key.char  # Get the character representation
    else:
        return str(key)  # Handle other cases


def on_press(key: Key | KeyCode):
    global is_recording
    now = time.time()

    if is_recording and key != exit_app_key:
        val = key_to_string(key)
        if val is None:
            return True

        if len(actions) and actions[-1]["a"] == "kd" and actions[-1]["val"] == val:
            # print(f"多次按下同一个键:{key}")
            return True

        action = {"a": "kd", "val": val, "t": now}
        actions.append(action)


def on_release(key):
    global is_recording
    now = time.time()

    if key == exit_app_key:
        winsound.Beep(frequency=1000, duration=1000)
        # 将 actions 保存到文件
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(actions, f)
        return False

    if key == Key.end:
        is_recording = not is_recording
        winsound.Beep(frequency=1000, duration=1000)
        return True

    if is_recording:
        val = key_to_string(key)
        if val is None:
            return True

        action = {"a": "ku", "val": val, "t": now}

        if (
            len(actions)
            and actions[-1]["a"] == "kd"
            and actions[-1]["val"] == val
            and now - actions[-1]["t"] < 0.18
        ):
            # print(f"{key}按下{actions[-1]["t"]}和抬起{now}的时间很短，作为点击")
            actions[-1]["a"] = "kp"
            actions[-1]["t"] = now
            return True
        actions.append(action)

    return True


def on_click(x, y, button, pressed):
    global is_recording
    now = time.time()

    print('point: ', x, y)

    if is_recording:
        if button == Button.left:
            if pressed:
                action = {"a": f"lmdown", "val": (x, y), "t": now}
                actions.append(action)
            else:
                if (
                    len(actions)
                    and actions[-1]["a"] == "lmdown"
                    and now - actions[-1]["t"] < 0.18
                ):
                    actions[-1]["a"] = "lmclick"
                    actions[-1]["t"] = now
                    return True
                action = {"a": f"lmup", "val": (x, y), "t": now}
                actions.append(action)
        elif button == Button.right:
            if pressed:
                action = {"a": f"rmdown", "val": (x, y), "t": now}
                actions.append(action)
            else:
                if (
                    len(actions)
                    and actions[-1]["a"] == "rmdown"
                    and now - actions[-1]["t"] < 0.18
                ):
                    actions[-1]["a"] = "rmclick"
                    actions[-1]["t"] = now
                    return True
                action = {"a": f"rmup", "val": (x, y), "t": now}
                actions.append(action)


# 监听键盘和鼠标
def listener_all():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        with pynput.mouse.Listener(on_click=on_click) as mouse_listener:
            while listener.running:
                if not is_recording:
                    time.sleep(1)
                    print("等待启动 end")
                    continue
                time.sleep(0.01)


def reverse_macro_all(file: str, sleep_seconds=0):
    if sleep_seconds:
        time.sleep(sleep_seconds)

    with open(file, "r", encoding="utf-8") as f:
        actions = json.load(f)

    becore_action = None
    for action in actions:
        duration = 0
        if becore_action is not None:
            seconds = action["t"] - becore_action["t"]
            duration = seconds
            if duration > 3:
                duration = random.uniform(2, 3)
            time.sleep(seconds)

        match action["a"]:
            case "kp":
                pydirectinput.press(action["val"])
            case "kd":
                pydirectinput.keyDown(action["val"])
            case "ku":
                pydirectinput.keyUp(action["val"])
            case "lmclick":
                pyautogui.moveTo(action["val"][0], action["val"][1], duration=duration)
                pyautogui.leftClick()
            case "rmclick":
                pyautogui.moveTo(action["val"][0], action["val"][1], duration=duration)
                pyautogui.leftClick()
            case "lmdown":
                pyautogui.moveTo(action["val"][0], action["val"][1], duration=duration)
                pyautogui.mouseDown(button=pyautogui.LEFT)
            case "lmup":
                pyautogui.mouseUp(button=pyautogui.LEFT)
            case "rmdown":
                pyautogui.moveTo(action["val"][0], action["val"][1], duration=duration)
                pyautogui.mouseDown(button=pyautogui.RIGHT)
            case "rmup":
                pyautogui.mouseUp(button=pyautogui.RIGHT)
        becore_action = action


if __name__ == "__main__":
    outfile = "北荒战云_前往天衡传送门.json"
    # listener_all()
    reverse_macro_all(outfile, 3)
