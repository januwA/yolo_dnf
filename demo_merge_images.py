import json
import os
import random
import cv2
import numpy as np
import glob
from PIL import Image


def merge_images(big_img_path, small_img_paths):
    """
    将多张小图合并到一张大图上

    Args:
        big_img_path: 大图路径
        small_img_paths: 小图路径列表
    """

    # 读取大图
    # big_img = cv2.imread(big_img_path)
    big_img = cv2.imdecode(np.fromfile(big_img_path, dtype=np.uint8), -1)
    h, w, c = big_img.shape

    # 计算每个小图在拼接图中的尺寸
    small_img_size = (w // 3, h // 2)

    # 创建一个空白的拼接图
    merged_img = np.zeros((h, w, c), dtype=np.uint8)

    # 将小图依次放入拼接图中
    row, col = 0, 0
    for i, small_img_path in enumerate(small_img_paths):
        small_img = cv2.imdecode(np.fromfile(small_img_path, dtype=np.uint8), -1)
        small_img = cv2.resize(small_img, small_img_size)
        small_img_rgb = cv2.cvtColor(small_img, cv2.COLOR_RGBA2RGB)
        merged_img[
            row * small_img_size[1] : (row + 1) * small_img_size[1],
            col * small_img_size[0] : (col + 1) * small_img_size[0],
        ] = small_img_rgb
        col += 1
        if col >= 3:
            row += 1
            col = 0

    # 显示并保存拼接图
    cv2.imshow("Merged Image", merged_img)
    cv2.waitKey(0)
    cv2.imwrite("merged.jpg", merged_img)


def merge_images_randomly(big_img_path, small_img_paths):
    """
    将多张小图随机分布到一张大图上

    Args:
        big_img_path: 大图路径
        small_img_paths: 小图路径列表
    """

    # 读取大图
    big_img = cv2.imdecode(np.fromfile(big_img_path, dtype=np.uint8), -1)
    h, w, c = big_img.shape

    # 读取小图并获取尺寸信息
    small_imgs = []
    small_img_sizes = []
    for path in small_img_paths:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        small_imgs.append(img)
        small_img_sizes.append(img.shape[:2])

    # 随机生成小图的位置和大小
    for img, size in zip(small_imgs, small_img_sizes):
        # 随机生成左上角坐标
        x = random.randint(0, w - size[1])
        y = random.randint(0, h - size[0])

        # 随机生成缩放比例
        scale = random.uniform(0.8, 1.2)
        new_size = (int(size[1] * scale), int(size[0] * scale))

        # 缩放小图
        resized_img = cv2.resize(img, new_size)

        # 将小图叠加到大图上
        roi = big_img[y : y + new_size[1], x : x + new_size[0]]

        resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_RGBA2RGB)
        cv2.addWeighted(resized_img_rgb, 0.5, roi, 0.5, 0, roi)

    # 显示并保存拼接图
    cv2.imshow("Merged Image", big_img)
    cv2.waitKey(0)
    cv2.imwrite("merged_random.jpg", big_img)


def generate_random_coordinates(mean=50, std=20, width=100, height=100):
    """
    生成在指定范围内符合正态分布的随机坐标

    Args:
        mean: 正态分布的均值
        std: 正态分布的标准差
        width: 坐标范围的宽度
        height: 坐标范围的高度

    Returns:
        tuple: 生成的随机坐标 (x, y)
    """

    x = int(np.random.randn() * std + mean)
    y = int(np.random.randn() * std + mean)

    x = np.clip(x, 0, width)
    y = np.clip(y, 0, height)

    return int(x), int(y)


out_dir = r"C:\Users\16418\Desktop\风暴幽城\segment_merge_player_enemy"
save_i = 1


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


def generate_rectangles_with_overlap(big_img, img, rectangles, overlap_threshold=0.3):
    """生成允许重叠的矩形"""
    y_pad = 124 # 小地图之下，这是在800x600的背景图上
    while True:
        # 生成随机矩形
        x = random.randint(0, big_img.width - img.width)
        y = random.randint(y_pad, big_img.height - img.height)
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


def merge_paste(map_img, img_paths, rect_list, labelme_file_data, label):
    imgs = [Image.open(path).convert("RGBA") for path in img_paths]
    for img in imgs:
        img_rect = generate_rectangles_with_overlap(map_img, img, rect_list)
        rect_list.append(img_rect)

        if random.randint(0, 1) == 1:
            # 左右翻转图像
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        map_img.paste(
            img, (img_rect[0], img_rect[1]), img
        )  # 使用paste方法，直接覆盖到大图上

        shape = {
            "label": label,  # 替换你的标签名
            "points": [img_rect[:2], img_rect[2:]],  # 矩形
            "group_id": None,
            "shape_type": "rectangle",  # 类型 矩形
            "flags": {},
        }
        labelme_file_data["shapes"].append(shape)


def merge_images_randomly_with_alpha(map_img_path, player_img_paths, enemy_img_paths):
    """
    将多张PNG小图随机分布到一张大图上，保持透明度

    Args:
        big_img_path: 大图路径
        small_img_paths: 小图路径列表
    """
    global save_i

    # 读取大图
    map_img = Image.open(map_img_path)
    map_img = map_img.convert("RGBA")  # 转换为RGBA模式
    
    out_img_name = f"{save_i}.png"
    out_img_filename = os.path.join(out_dir, out_img_name)

    labelme_file_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": out_img_name,  # 替换为你的图片路径
        "imageData": None,
        "imageHeight": map_img.height,  # 替换为你的图片高度
        "imageWidth": map_img.width,  # 替换为你的图片宽度
    }

    rect_list = []
    merge_paste(map_img, enemy_img_paths, rect_list, labelme_file_data, "敌人")
    merge_paste(map_img, player_img_paths, rect_list, labelme_file_data, "玩家")

    # 保存图像
    map_img.save(out_img_filename, "PNG")

    # 文件不存在会自动创建
    with open(os.path.join(out_dir, f"{save_i}.json"), "w", encoding="utf-8") as f:
        json.dump(labelme_file_data, f, indent=2)
    save_i += 1


def merge_image_to_mapimage(map_images, player_images, enemy_images):
    """每张地图背景上生成8个敌人, 然后换下一张背景图, 知道小图贴完"""

    # 打乱素材
    random.shuffle(player_images)
    random.shuffle(enemy_images)

    map_len = len(map_images)
    player_len = len(player_images)
    enemy_len = len(enemy_images)

    enemy_step = 6  # 每次多少个敌人
    player_step = 4  # 每次多少个玩家

    i = 0
    player_begin = 0
    enemy_begin = 0
    count = 0
    while True:
        if count >= 2:
            break

        if enemy_begin >= enemy_len:
            print("敌人已经循环结束了")
            count += 1
            enemy_begin = 0

        if player_begin >= player_len:
            print("玩家已经循环结束了")
            count += 1
            player_begin = 0

        if i >= map_len:
            i = 0

        enemy_end = enemy_begin + enemy_step
        if enemy_end > enemy_len:
            enemy_end = enemy_len

        player_end = player_begin + player_step
        if player_end > player_len:
            player_end = player_len

        merge_images_randomly_with_alpha(
            map_images[i],
            player_images[player_begin : player_begin + player_step],
            enemy_images[enemy_begin : enemy_begin + enemy_step],
        )
        i += 1

        enemy_begin = enemy_begin + enemy_step
        player_begin = player_begin + player_step


if __name__ == "__main__":
    # 所有玩家素材
    player_images = glob.glob(
        os.path.join(r"C:\Users\16418\Desktop\风暴幽城\玩家", "*/*/*.png")
    )

    # 所有敌人素材
    enemy_images = glob.glob(
        os.path.join(r"C:\Users\16418\Desktop\风暴幽城\敌人", "*/*/*.png")
    )

    # 地图背景素材
    map_images = glob.glob(
        os.path.join(r"C:\Users\16418\Desktop\风暴幽城\截图", "*.jpg")
    )
    random.shuffle(player_images)
    random.shuffle(enemy_images)
    merge_image_to_mapimage(map_images, player_images, enemy_images)
