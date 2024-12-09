import os
import cv2
import numpy as np
import glob

images_dir = r"C:\Users\16418\Desktop"
images = glob.glob(os.path.join(images_dir, "*.png"))
image_list = []
if __name__ == "__main__":
    # print(images)
    for p in images:
        i = cv2.imread(p)
        image_list.append(i)

    # cv2.imshow("result", image_list[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(0.1)  # Adjust this threshold as needed
    status, pano = stitcher.stitch(image_list)
    if status == cv2.Stitcher_OK:
        cv2.imwrite("o.jpg", pano)
        cv2.imshow("result", pano)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("可供拼接的图像数量不足")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("单应性估计失败")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("相机参数调整失败")
    else:
        print("Unknown error")
