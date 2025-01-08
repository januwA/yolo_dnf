# 添加副本地图

import json
import os
import random
import cv2
import numpy as np
import glob
from PIL import Image


if __name__ == "__main__":
    jps = glob.glob(os.path.join(r"C:\Users\16418\Desktop\LiuYuPuBu\am_2", "*.json"))
    for jp in jps:
        with open(jp, "r+", encoding="utf-8") as fp:
            jd = json.load(fp)
            if jd and jd["shapes"] is not None:
                jd["shapes"].append(
                    {
                        "label": "副本地图",
                        "points": [[1460, 48], [1592, 112]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                    },
                )
                fp.seek(0)
                json.dump(jd, fp, indent=2)
