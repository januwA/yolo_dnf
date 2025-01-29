import ctypes
import time
import cv2
import win32gui
import win32api
import win32ui
import win32print
import win32con
import numpy as np
from PIL import Image, ImageGrab

from demo_matchimg import find_template
from g_dnf import window_capture

# hwnd = win32gui.GetForegroundWindow()
hwnd = 133252

template_path = r"C:\Users\16418\Desktop\dnf_py\image_template\副本中.png"
template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)

if hwnd:
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    while True:
        img = window_capture(hwnd, toCv2=True, usePIL=False)
        
        img = find_template(img, template, 0.4)

        # win_pos = win32gui.GetWindowRect(hwnd)
        # game_frame = ImageGrab.grab(bbox=win_pos)
        # img = cv2.cvtColor(np.array(game_frame), cv2.COLOR_RGB2BGR)
        
        cv2.resizeWindow("test", 700, 400)
        cv2.imshow("test", img)
        cv2.waitKey(5)
else:
    print("窗口未找到")
