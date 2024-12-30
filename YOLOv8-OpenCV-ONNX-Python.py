# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2.dnn
import numpy as np
from pypinyin import pinyin, Style

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

# CLASSES = yaml_load(check_yaml(r"C:\Users\16418\Desktop\FenBaoYouChen\segment_merge4_1\YOLODataset\dataset.yaml"))["names"]
CLASSES = {0: 'wanjia', 1: 'fubenditu', 2: 'cailiao', 3: 'men', 4: 'diren', 5: 'jiangli'}
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def chinese_to_pinyin(text):
  """将中文文本转换为不带声调的拼音

  Args:
    text: 中文文本

  Returns:
    一个包含拼音列表的列表
  """

  result = []
  for p in pinyin(text, style=Style.NORMAL):
    result.append(''.join(p))
  return result

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    # "_".join(chinese_to_pinyin())
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    """
    Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
    """
    # Load the ONNX model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    print(height, width)

    # 准备一个方形图像用于推理
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640
    print(f"{scale=}")

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()
    
    # 准备输出数组
    outputs = np.array([cv2.transpose(outputs[0])])
    # outputs.shape=(1, 8400, 10)
    print( outputs.shape )
    rows = outputs.shape[1] # 8400

    boxes = []
    scores = []
    class_ids = []

    # 迭代输出以收集边界框、置信度分数和类别 ID
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        # print(outputs[0][i][:4], classes_scores )
        # break
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        print(box)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # Display the image with bounding boxes
    cv2.imshow("image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections

# python .\YOLOv8-OpenCV-ONNX-Python.py --model "C:\Users\16418\Desktop\FenBaoYouChen\Config\best.onnx" --img "C:\Users\16418\Desktop\FenBaoYouChen\segment_merge4\1.png"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    args = parser.parse_args()
    main(args.model, args.img)
