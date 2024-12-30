import cv2
import numpy as np


def find_template(img, template, threshold=0.25):
    img_h, img_w = img.shape[:2]
    template_h, template_w = template.shape[:2]

    # 指定搜索区域的左上角坐标和右下角坐标
    # search_x, search_y = img_w - 300, 200
    # search_x, search_y = 0, img_h
    # search_region = img[0:search_y, search_x:-1]  # 裁剪

    # cv2.rectangle(img, (search_x, 0), (img_w, search_y), (0, 0, 255), 2)

    # # 转换为灰度图 COLOR_BGR2GRAY
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    # # 模板匹配
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    # print(res)  # 匹配度

    # # 设置阈值，找到所有大于阈值的匹配
    loc = np.where(res >= threshold)

    result = list(zip(*loc[::-1]))
    print(f"匹配到: {len(result)}个")

    # 绘制矩形框
    # for pt in zip(*loc[::-1]):
    #     # for pt in result:
    #     x, y = pt
    #     x += search_x
    #     cv2.rectangle(img, (x, y), (x + template_w, y + template_h), (0, 0, 255), 2)

    return img


if __name__ == "__main__":
    # image_path = r"C:\Users\16418\Desktop\FenBaoYouChen\segment\1_00000109.jpg"
    image_path = r"C:\Users\16418\Desktop\dnf_py\1_5.png"
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img = cv2.resize(img, (40, 40))

    template_path = r"C:\Users\16418\Desktop\dnf_py\image_template\副本中.png"
    template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)

    result_img = find_template(img, template, 0.8)

    # 显示结果
    cv2.imshow("Detected", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
