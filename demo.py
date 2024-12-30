import cv2
import numpy as np
from PIL import Image

if __name__ == "__main__":
  # 读取图像
  img = Image.open(r"C:\Users\16418\Pictures\1cf7da9749d21854fc83699635f9e15.jpg")

  # PIL 转换为 OpenCV:
  img=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  
  # OpenCV 转换为 PIL:
  # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  
  # img = cv2.imread(r"C:\Users\16418\Pictures\1cf7da9749d21854fc83699635f9e15.jpg")

  # 定义亮度和对比度系数
  alpha = 1  # 对比度控制，大于1增加对比度，小于1降低对比度

  # 亮度控制，正值增加亮度，负值降低亮度
  for beta in range(-100, 100, 20):
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imshow(f'New Image: {beta}', new_img)

  cv2.waitKey(0)
  cv2.destroyAllWindows()
