import getopt
import os
import sys
import cv2
import glob
import json
import numpy as np
from PIL import Image

fname_split = "_beta_"


def remove_all_beta_file():
    """删除所有生成的图片"""
    ps = glob.glob(
        os.path.join(
            r"C:\Users\16418\Desktop\FenBaoYouChen\auto_mark_test", f"*{fname_split}*.*"
        )
    )
    for p in ps:
        os.remove(p)


def make_beta_file():
    jf_paths = glob.glob(
        os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\auto_mark_test", "*.json")
    )
    for jfp in jf_paths:
        jfp_dir = os.path.dirname(jfp)
        with open(jfp, encoding="utf-8") as f:
            data = json.load(f)
            imagePath = data["imagePath"]
            if not imagePath:
                continue
            if not os.path.isabs(imagePath):
                imagePath = os.path.join(jfp_dir, imagePath)

            img = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), -1)

            # 'cat' '.jpg'
            (fname, fext) = os.path.splitext(os.path.basename(imagePath))

            # 对比度控制，大于1增加对比度，小于1降低对比度
            alpha = 1

            # 亮度控制，正值增加亮度，负值降低亮度
            for beta in range(-200, 200, 50):
                if beta == 0:
                    continue
                # new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

                # 正变暗，负变亮
                new_img = cv2.subtract(img, beta)

                # 写入新图片和json
                new_img_name = f"{fname}{fname_split}{beta}{fext}"
                new_img_path = os.path.join(jfp_dir, new_img_name)
                new_json_path = os.path.join(
                    jfp_dir, f"{fname}{fname_split}{beta}.json"
                )
                cv2.imwrite(new_img_path, new_img)
                with open(new_json_path, "w", encoding="utf-8") as fj:
                    data["imagePath"] = new_img_name
                    json.dump(data, fj, indent=2)


def main():
    _, args = getopt.getopt(sys.argv[1:], "")
    print(args)
    if len(args) == 0:
        return

    if args[0] == "r" or args[0] == "remove":
        print("删除")
        remove_all_beta_file()
    if args[0] == "m" or args[0] == "make":
        print("生成")
        make_beta_file()


if __name__ == "__main__":
    main()

    # 读取图像
    # img = Image.open(r"C:\Users\16418\Pictures\1cf7da9749d21854fc83699635f9e15.jpg")

    # PIL 转换为 OpenCV:
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # OpenCV 转换为 PIL:
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # img = cv2.imread(r"C:\Users\16418\Pictures\1cf7da9749d21854fc83699635f9e15.jpg")
    # reduced_brightness = cv2.subtract(img, -100)

    # cv2.imshow("Original", img)
    # cv2.imshow("Reduced Brightness", reduced_brightness)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 定义亮度和对比度系数
    alpha = 1  # 对比度控制，大于1增加对比度，小于1降低对比度

    # 亮度控制，正值增加亮度，负值降低亮度
    # for beta in range(-100, 100, 20):
    #     new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    #     cv2.imshow(f"New Image: {beta}", new_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
