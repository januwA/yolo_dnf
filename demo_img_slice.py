import cv2
import numpy as np


def imread2(p: str) -> cv2.typing.MatLike:
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), -1)
    return img


if __name__ == "__main__":
    img = imread2(r"C:\Users\16418\Desktop\微信截图_20250216144200.jpg")
    h, w = img.shape[:2]
    img = img[int(h * 0.05) : int(h * 0.15), int(w * 0.3) : int(w * 0.7)]
    cv2.imshow("img", img)
    cv2.waitKey(0)
