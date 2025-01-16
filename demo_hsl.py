import time
import cv2
import numpy as np


def shift_channel(c, amount):
    if amount > 0:
        lim = 255 - amount
        c[c >= lim] = 255
        c[c < lim] += amount
    elif amount < 0:
        amount = -amount
        lim = amount
        c[c <= lim] = 0
        c[c > lim] -= amount
    return c


title = "滑块"
cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(title, 350, 700)


def nothing(val):
    pass


cv2.createTrackbar("HMin", title, 0, 179, nothing)
cv2.createTrackbar("SMin", title, 0, 255, nothing)
cv2.createTrackbar("VMin", title, 0, 255, nothing)
cv2.createTrackbar("HMax", title, 0, 179, nothing)
cv2.createTrackbar("SMax", title, 0, 255, nothing)
cv2.createTrackbar("VMax", title, 0, 255, nothing)

# 设置默认的 HSV Max
cv2.setTrackbarPos("HMax", title, 179)
cv2.setTrackbarPos("SMax", title, 255)
cv2.setTrackbarPos("VMax", title, 255)

cv2.createTrackbar("SAdd", title, 0, 255, nothing)
cv2.createTrackbar("SSub", title, 0, 255, nothing)
cv2.createTrackbar("VAdd", title, 0, 255, nothing)
cv2.createTrackbar("VSub", title, 0, 255, nothing)

img = cv2.imread(r"C:\Users\16418\Desktop\hsl.jpg")

# 色相、饱和度、明度
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("rgb", img)

while True:
    HMin = cv2.getTrackbarPos("HMin", title)
    SMin = cv2.getTrackbarPos("SMin", title)
    VMin = cv2.getTrackbarPos("VMin", title)
    HMax = cv2.getTrackbarPos("HMax", title)
    SMax = cv2.getTrackbarPos("SMax", title)
    VMax = cv2.getTrackbarPos("VMax", title)
    SAdd = cv2.getTrackbarPos("SAdd", title)
    SSub = cv2.getTrackbarPos("SSub", title)
    VAdd = cv2.getTrackbarPos("VAdd", title)
    VSub = cv2.getTrackbarPos("VSub", title)

    # 转换原图
    h, s, v = cv2.split(hsv)
    s = shift_channel(s, SAdd)
    s = shift_channel(s, -SSub)
    v = shift_channel(v, VAdd)
    v = shift_channel(v, -VSub)
    hsv = cv2.merge([h, s, v])

    lower = np.array([HMin, SMin, VMin])
    upper = np.array([HMax, SMax, VMax])

    # 在图像中提取特定范围内的像素值
    # 输入: 一幅图像（通常是 HSV 格式）、一个下限值数组和一个上限值数组。
    # 输出: 一幅二值图像，满足条件的像素点为白色（255），不满足条件的像素点为黑色（0）。
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("mask", mask)

    # 与一个掩模进行按位与操作,对应位置的比特位都为 1 时，结果的该位才为 1，否则为 0
    # 提取出对应的颜色，其它全部为黑色
    result = cv2.bitwise_and(hsv, hsv, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    cv2.imshow("result", result)
    cv2.waitKey(1)
    # cv2.imshow("hsl", hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
