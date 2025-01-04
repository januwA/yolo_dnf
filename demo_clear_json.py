"""
清理空的json，没有图片
"""

import os
import sys
import cv2
import glob
import json

if __name__ == "__main__":
    jf_paths = glob.glob(
        # os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\截图", "*/*.json")
        os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\segment_merge4_1", "*.json")
    )

    need_removes = []
    for jfp in jf_paths:
        jfp_dir = os.path.dirname(jfp)
        with open(jfp, encoding="utf-8") as f:
            data = json.load(f)
            imagePath = data["imagePath"]
            if not imagePath:
                need_removes.append(jfp)
                continue

            if not os.path.isabs(imagePath):
                imagePath = os.path.join(jfp_dir, imagePath)

            if not os.path.exists(imagePath):
                need_removes.append(jfp)

    print("need_removes", len(need_removes))
    for p in need_removes:
        os.remove(p)
