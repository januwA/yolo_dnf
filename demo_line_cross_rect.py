import math


def is_intersect(line, rect):
    """
    判断线段是否与矩形相交

    Args:
      line: 线段，表示为两个点的列表，如 [(0, 0), (100, 100)]
      rect: 矩形，表示为左下角和右上角坐标的列表，如 [(50, 50), (70, 70)]

    Returns:
      bool: True表示相交，False表示不相交
    """

    # 获取线段的两个端点和矩形的四个顶点
    x1, y1 = line[0]
    x2, y2 = line[1]
    rect_x1, rect_y1, rect_x2, rect_y2 = rect[:4]
    # rect_x1, rect_y1 = rect[0]
    # rect_x2, rect_y2 = rect[1]

    # 判断线段的两个端点是否在矩形内
    if (rect_x1 <= x1 <= rect_x2 and rect_y1 <= y1 <= rect_y2) or (
        rect_x1 <= x2 <= rect_x2 and rect_y1 <= y2 <= rect_y2
    ):
        return True

    # 判断线段是否与矩形的四条边相交
    # 这里使用向量叉积判断线段是否与矩形的边相交
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    # 矩形的四个顶点
    rect_points = [
        (rect_x1, rect_y1),
        (rect_x2, rect_y1),
        (rect_x2, rect_y2),
        (rect_x1, rect_y2),
    ]
    for i in range(4):
        if (
            ccw(line[0], line[1], rect_points[i])
            * ccw(line[0], line[1], rect_points[(i + 1) % 4])
            <= 0
            and ccw(rect_points[i], rect_points[(i + 1) % 4], line[0])
            * ccw(rect_points[i], rect_points[(i + 1) % 4], line[1])
            <= 0
        ):
            return True

    return False


if __name__ == "__main__":
    line = [(0, 0), (50, 0)]
    # rect = [(50, 50), (70, 70)]
    rect = [51, -10, 55, 12]
    print(is_intersect(line, rect))
