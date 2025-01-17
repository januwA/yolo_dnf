import json
import re
import time
import pyautogui
import pynput
from pynput.keyboard import Key, Listener, KeyCode
from pynput.mouse import Button, Controller

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
    if is_recording and key != exit_app_key:
        val = key_to_string(key)
        if val is None:
            return True

        和上次一样 = False
        action = {"a": "kd", "val": val, "t": time.time()}
        if (
            len(actions)
            and actions[-1]["a"] == action["a"]
            and actions[-1]["val"] == action["val"]
        ):
            和上次一样 = True

        if not 和上次一样:
            actions.append(action)


def on_release(key):
    global is_recording
    if key == exit_app_key:
        # 将 actions 保存到文件
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(actions, f)
        return False

    if key == Key.end:
        is_recording = not is_recording
        return True

    if is_recording:
        val = key_to_string(key)
        if val is None:
            return True

        action = {"a": "ku", "val": val, "t": time.time()}
        actions.append(action)

    return True


def on_click(x, y, button, pressed):
    global is_recording

    k = "down" if pressed else "up"
    if is_recording:
        if button == Button.left:
            action = {"a": f"lm{k}", "val": (x, y), "t": time.time()}
            actions.append(action)
        elif button == Button.right:
            action = {"a": f"rm{k}", "val": (x, y), "t": time.time()}
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


def reverse_all(file: str, sleep_seconds=0):
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
            if seconds < 0.1:
                seconds = 0.1
            time.sleep(seconds)

        match action["a"]:
            case "kp":
                pyautogui.press(action["val"])
            case "kd":
                pyautogui.keyDown(action["val"])
            case "ku":
                pyautogui.keyUp(action["val"])
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
    outfile = "actions2.json"
    listener_all()
    # reverse_all(outfile, 3)
