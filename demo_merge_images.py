import json
import os
import random
import cv2
import numpy as np
import glob
from PIL import Image

label = {
    "wanjia": "玩家",
    "diren": "敌人",
    "men": "门",
    "cailiao": "材料",
    "jaingli": "奖励",
    "sailiya": "赛利亚",
    "youxikaishi": "游戏开始",
    "fubenditu": "副本地图",
    "xuanzeditu": "选择地图",
}

out_dir = r"C:\Users\16418\Desktop\FenBaoYouChen\segment_merge4"
save_i = 1


def jsonpath_from_imagepath(imagepath: str):
    fname = os.path.splitext(os.path.basename(imagepath))[0]
    jsonpath = os.path.join(os.path.dirname(imagepath), f"{fname}.json")
    return jsonpath


def calculate_overlap(rect1, rect2):
    """计算两个矩形的重叠面积"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 计算相交矩形的左上角和右下角坐标
    x_overlap = max(x1, x2)
    y_overlap = max(y1, y2)
    x_end = min(x1 + w1, x2 + w2)
    y_end = min(y1 + h1, y2 + h2)

    # 如果两个矩形不相交，返回0
    if x_end <= x_overlap or y_end <= y_overlap:
        return 0

    # 计算相交矩形的面积和并集的面积
    intersection_area = (x_end - x_overlap) * (y_end - y_overlap)
    union_area = w1 * h1 + w2 * h2 - intersection_area

    # 计算重叠比例
    overlap_ratio = intersection_area / union_area
    return overlap_ratio


def generate_rectangles_with_overlap(
    big_img, img, rectangles, smap_rect: list[float], overlap_threshold=0.3
):
    """生成允许重叠的矩形"""
    状态栏高度 = 62

    while True:
        # 生成随机矩形
        x = random.randint(0, big_img.width - img.width)
        y = random.randint(int(smap_rect[3]), big_img.height - 状态栏高度 - img.height)
        w = x + img.width
        h = y + img.height
        new_rect = [x, y, w, h]

        # 检查新矩形与已有的矩形是否重叠超过阈值
        overlap = False
        for rect in rectangles:
            overlap_ratio = calculate_overlap(new_rect, rect)
            if overlap_ratio > overlap_threshold:
                overlap = True
                break

        if not overlap:
            return new_rect


def merge_paste(
    map_img,
    img_paths: list[str],
    rect_list,
    labelme_file_data,
    label: str,
    smap_rect: list[float],
):
    imgs = [Image.open(path).convert("RGBA") for path in img_paths]
    for img in imgs:
        img_rect = generate_rectangles_with_overlap(map_img, img, rect_list, smap_rect)
        rect_list.append(img_rect)

        if random.randint(0, 1) == 1:
            # 左右翻转图像
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        map_img.paste(
            img, (img_rect[0], img_rect[1]), img
        )  # 使用paste方法，直接覆盖到大图上

        labelme_file_data["shapes"].append(
            {
                "label": label,  # 替换你的标签名
                "points": [img_rect[:2], img_rect[2:]],  # 矩形
                "group_id": None,
                "shape_type": "rectangle",  # 类型 矩形
                "flags": {},
            }
        )


def merge_images_randomly_with_alpha(map_img_path, smap_rect, lst):
    """
    将多张PNG小图随机分布到一张大图上
    """
    global save_i

    # 读取大图
    map_img = Image.open(map_img_path)
    map_img = map_img.convert("RGBA")  # 转换为RGBA模式

    out_img_name = f"{save_i}.png"
    out_img_filename = os.path.join(out_dir, out_img_name)
    labelme_file_data = None

    map_img_jsonpath = jsonpath_from_imagepath(map_img_path)

    if os.path.exists(map_img_jsonpath):
        with open(map_img_jsonpath, "r", encoding="utf-8") as f:
            labelme_file_data = json.load(f)
    else:
        labelme_file_data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": out_img_name,  # 替换为你的图片路径
            "imageData": None,
            "imageHeight": map_img.height,  # 替换为你的图片高度
            "imageWidth": map_img.width,  # 替换为你的图片宽度
        }

    labelme_file_data["imagePath"] = out_img_name

    rect_list: list[list[float, float, float, float]] = []

    # 先把已存在的矩形添加进去，避免被覆盖太多
    for el in labelme_file_data["shapes"]:
        if el["points"] and type(el["points"]) is list:
            rect_list.append(np.array(el["points"]).flatten().tolist())

    # 添加小地图标记
    labelme_file_data["shapes"].append(
        {
            "label": label["fubenditu"],
            "points": [smap_rect[:2], smap_rect[2:]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }
    )

    for el in lst:
        merge_paste(
            map_img,
            el["images_split"],
            rect_list,
            labelme_file_data,
            el["label"],
            smap_rect,
        )

    # 保存图像
    map_img.save(out_img_filename, "PNG")

    # 文件不存在会自动创建
    with open(os.path.join(out_dir, f"{save_i}.json"), "w", encoding="utf-8") as f:
        json.dump(labelme_file_data, f, indent=2)
    save_i += 1


def merge_image_to_mapimage():
    """每张地图背景上生成8个敌人, 然后换下一张背景图, 知道小图贴完"""

    # 地图背景素材 800x600
    map_image_list = [
        {
            "images": glob.glob(
                os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\截图\魔道", "*.jpg")
            ),
            "smap_rect": None,
        },
        {
            "images": glob.glob(
                os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\截图\元素", "*.jpg")
            ),
            "smap_rect": None,
        },
        {
            "images": glob.glob(
                os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\截图\枪炮", "*.jpg")
            ),
            "smap_rect": None,
        },
        {
            "images": glob.glob(
                os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\截图\枪炮2", "*.jpg")
            ),
            "smap_rect": None,
        },
    ]
    for el in map_image_list:
        if el["smap_rect"] is None:
            print(f"地图 素材量:{len(el["images"])}")
            # 尝试从json中提取一次小地图的定位
            for imgp in el["images"]:
                ok = False
                jsonpath = jsonpath_from_imagepath(imgp)
                if not os.path.exists(jsonpath):
                    continue
                with open(jsonpath, "r", encoding="utf-8") as fp:
                    jd = json.load(fp)
                    if jd and jd["shapes"] and type(jd["shapes"]) is list:
                        for shape in jd["shapes"]:
                            if shape["label"] == label["fubenditu"]:
                                # 设置为一维数组
                                el["smap_rect"] = tuple(
                                    np.array(shape["points"]).flatten().tolist()
                                )
                                ok = True

                if ok:
                    break

    # print([el["smap_rect"] for el in map_image_list])
    # return

    lst = [
        # 所有玩家素材
        {
            "images": glob.glob(
                os.path.join(r"C:\Users\16418\Desktop\dnf素材\玩家皮肤", "*/*/*.png")
            ),
            "label": label["wanjia"],
            "step": 1,
            "begin": 0,
            "end": 0,
            "images_split": [],
            "success": False,
        },
        # 所有敌人素材
        {
            "images": glob.glob(
                os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\敌人", "*/*/*.png")
            ),
            "label": label["diren"],
            "step": 3,
            "begin": 0,
            "end": 0,
            "images_split": [],
            "success": False,
        },
        # 所有门素材
        # {
        #     "images": glob.glob(
        #         os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\门", "*.png")
        #     ),
        #     "label": label["men"],
        #     "step": 2,
        #     "begin": 0,
        #     "end": 0,
        #     "images_split": [],
        #     "success": False,
        # },
        # 所有奖励素材
        {
            "images": glob.glob(
                os.path.join(r"C:\Users\16418\Desktop\FenBaoYouChen\奖励", "*.png")
            ),
            "label": label["jaingli"],
            "step": 1,
            "begin": 0,
            "end": 0,
            "images_split": [],
            "success": False,
        },
    ]

    # 打乱素材
    最大素材量 = 2000
    for el in lst:
        random.shuffle(el["images"])
        print(f"{el['label']} 素材量:{len(el["images"])}")
        if len(el["images"]) > 最大素材量:
            el["images"] = el["images"][:最大素材量]

    map_image_list_i = 0
    map_images_i = 0
    count = 0

    while True:
        map_image = map_image_list[map_image_list_i]

        # 全部完成
        if count >= len(lst):
            break

        for el in lst:
            img_len = len(el["images"])
            if el["begin"] >= img_len:
                # print(f"{el['label']}已经循环结束了")
                el["begin"] = 0
                if not el["success"]:
                    el["success"] = True
                    count += 1
                    print(f"{el['label']} 已完成")

            el["end"] = el["begin"] + el["step"]
            if el["end"] > img_len:
                el["end"] = img_len

            el["images_split"] = el["images"][el["begin"] : el["end"]]

        merge_images_randomly_with_alpha(
            map_image["images"][map_images_i], map_image["smap_rect"], lst
        )

        map_images_i += 1
        if map_images_i >= len(map_image["images"]):
            map_image_list_i += 1
            map_images_i = 0
            if map_image_list_i >= len(map_image_list):
                map_image_list_i = 0

        for el in lst:
            el["begin"] = el["end"]


if __name__ == "__main__":
    if os.path.exists(out_dir):
        merge_image_to_mapimage()
    else:
        print(f"目录不存在: {out_dir}")
