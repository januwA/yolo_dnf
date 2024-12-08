## 添加ahk环境变量

> 项目目录不要到处移动，移动后py venv虚拟环境会出问题

## ffmpeg提取视频帧
```sh
ffmpeg -i "1.mp4" -r 1/3 -ss 00:00:00 -vf scale=1280:720 -q:v 7 -f image2 "segment\1_%8d.jpg"

ffmpeg -i "C:\Users\16418\Desktop\风暴幽城\剑魂 风暴幽城 普通 2024-12-02.mp4" -r 1 -ss 00:00:00 -vf scale=1280:720 -q:v 7 -f image2 "C:\Users\16418\Desktop\风暴幽城\segment\1_%8d.jpg"
```

- -r 1 输出帧速率，每5秒提取一帧 -r 1/5
- -ss 00:00:00 从0秒开始提取
- -vf scale=1280:720 调整输出分辨率，不然文件太大，scale=(iw/2):(ih/2)
- -q:v 7 有损压缩编码的质量,值越大压缩率越高，对PNG无效

## 安装yolo

### 如果要使用GPU，提前安装这些

nvidia-smi.exe 路径 "C:\Windows\System32\DriverStore\FileRepository\nvdm.inf_amd64_53dbfa2d292c9e64\nvidia-smi.exe" 运行后查看 CUDA Version: 12.5 

1. 安装 vc c++
2. nvidia CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit-archive  下载12.5或之下
3. nvidia cuDNN: https://developer.nvidia.com/rdp/cudnn-archive 选择符合CUDA的版本 Download cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x

安装 nvidia CUDA Toolkit 安装好后，将nvidia cuDNN压缩包的文件移动到 nvidia CUDA Toolkit 安装目录下，然后设置path变量

- D:\apps\NVIDIA_CUDA_TOOLKIT_12_5\bin
- D:\apps\NVIDIA_CUDA_TOOLKIT_12_5\include
- D:\apps\NVIDIA_CUDA_TOOLKIT_12_5\lib


### yolo

在 https://pytorch.org/ 可以找到你需要安装哪个版本的 cu*

```sh
pip install ultralytics
pip install supervision labelme labelme2yolo huggingface_hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 标注

- labelme 本地标注工具
- labelme2yolo --json_dir labelme_out_dir 将labelme输出的配置文件生成为数据集


运行测试:
> D:\apps\NVIDIA_CUDA_TOOLKIT_12_5\extras\demo_suite\bandwidthTest.exe
>
> D:\apps\NVIDIA_CUDA_TOOLKIT_12_5\extras\demo_suite\deviceQuery.exe

### 验证GPU是否可用

```py
import torch

print(torch.cuda.is_available())  # 输出True表示GPU可用
```

## 训练模型

训练模型的目的是产出best.pt模型

```sh
yolo task=detect mode=train model=yolov8n.pt data=YOLODataset\dataset.yaml epochs=100 imgsz=640 device=0

yolo task=detect mode=train model="C:\Users\16418\Desktop\风暴幽城\trains\train3\weights\best.pt" data="C:\Users\16418\Desktop\风暴幽城\segment_guiQi\YOLODataset\dataset.yaml" epochs=100 imgsz=640 device=0 project="C:\Users\16418\Desktop\风暴幽城\trains"
```

- task=detect: 指定任务为目标检测
- mode=train: 表示当前模式为训练模式，即用数据集训练模型
- model=yolov8s.pt 第一次会从网上下载，当本地存在时使用绝对路径
  - yolov8n: 参数量最小，速度最快，适用于对实时性要求高、计算资源有限的场景
  - yolov8s: 参数量适中，速度较快，在精度和速度之间取得了较好的平衡。
  - yolov8m: 参数量较大，精度较高，适用于对精度要求较高的场景。
  - yolov8l: 参数量最大，精度最高，适用于对精度要求极高的场景，但计算量也最大。
  - yolov8x: 参数量极大，精度极高，但计算量巨大，通常需要高性能的硬件设备。
- data=dataset/data.yaml: 指定数据集的配置文件。这个YAML文件包含了训练集、验证集的路径、类别信息等。
- epochs=100: 设置训练的总轮数为100轮
- imgsz=640: 设置输入图像的尺寸为640x640像素
- device=0 指定使用哪个GPU进行检测
- project=目录  指定了输出根目录


## 测试模型

使用训练好的best.pt模型

```sh
yolo task=detect mode=predict model="C:\Users\16418\Desktop\dnf_py\runs\detect\train\weights\best.pt" conf=0.25 source="C:\Users\16418\Desktop\shenDianWaiWei\1.mp4" device=0
```

- task=detect 明确指定任务是目标检测
- mode=predict 指示模型处于预测模式, 即使用训练好的模型对新的数据进行推理。
- model=yolov8s.pt 指定了要使用的模型文件路径
- conf=0.25: 设置了置信度阈值。只有检测到的目标的置信度大于0.25时才会被保留
- source=1.mp4 要进行检测的视频文件路径或图片目录
- device=0 指定使用哪个GPU进行检测


---


```py

# 将推理结果转换为 supervision.Detections 对象
detections = sv.Detections.from_ultralytics(result)
pprint.pprint(detections)

oriented_box_annotator = sv.OrientedBoxAnnotator()

annotated_frame = oriented_box_annotator.annotate(
    # scene=cv2.imread("3.jpg"),
    scene=cv2.imread(im1),
    detections=detections,
)

annotated_frame = sv.resize_image(annotated_frame, keep_aspect_ratio=True)
sv.cv2_to_pillow(annotated_frame)

# 创建一个 BoxAnnotator 对象，用于绘制边框，并设置边框厚度为 4
annotator = sv.BoxAnnotator(thickness=2)

# 使用 annotator 对象在图像上绘制检测到的目标的边框
annotated_image = annotator.annotate(im1, results)

# 创建一个 LabelAnnotator 对象，用于添加标签，并设置标签字体大小为 2 和厚度为 2
annotator = sv.LabelAnnotator(text_scale=1, text_thickness=1)

# 使用 annotator 对象在图像上添加检测到的目标的标签
annotated_image = annotator.annotate(annotated_image, results)

# 使用 plot_image 函数显示标注后的图像
sv.plot_image(annotated_image)
```


---

## 测试玩家移速：

### 测试1秒结果

注意不要让屏幕移动

```
按下: right
抬起: right
移速:250.12197024651792

点击: right
按下: right
抬起: right
移速:508.35519078691425

按下: right
按下: up
抬起: right
抬起: up
移速:298.75073221667594

按下: right
抬起: right
移速:241.0

按下: right
抬起: right
移速:243.0514348857048

按下: down
抬起: down
移速:170.0

按下: up
抬起: up
移速:173.0028901492689
```

```
player:0,0  boss:0,3
player:1,0  boss:1,5
```

## 标注文件json

```sh
labelme apc2016_obj3.jpg  # 指定图像文件
labelme apc2016_obj3.jpg -O apc2016_obj3.json  # 保存后关闭窗口
labelme apc2016_obj3.jpg --nodata  # JSON文件中不包含图像数据，但包含相对图像路径
labelme apc2016_obj3.jpg --labels 玩家,敌人,材料  # 指定标签列表

labelme data_annotated/  # 打开目录对其中的所有图像进行注释
labelme data_annotated/ --labels labels.txt  # 使用文件指定标签列表
```

```json
{
  "version": "5.5.0",
  "flags": {},
  "shapes": [
    {
      "label": "门",
      "points": [
        [
          1042.6829268292684,
          409.0243902439024
        ],
        [
          1150.9756097560976,
          533.4146341463414
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
    {
      "label": "玩家",
      "points": [
        [
          939.7560975609757,
          378.2926829268293
        ],
        [
          1039.7560975609756,
          521.2195121951219
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    }
  ],
  "imagePath": "1_00000147.jpg",
  "imageData": null,
  "imageHeight": 720,
  "imageWidth": 1280
}
```
