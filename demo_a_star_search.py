import heapq


def heuristic(node, goal):
    # 曼哈顿距离作为启发函数
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def a_star_search(start, goal, grid, 此路不通):
    open_set = [(heuristic(start, goal), start)]
    # print(open_set)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]
        # print(current)

        # 到达目标
        if current == goal:
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]  # pre current point
            path.insert(0, start)
            return path

        # 扩散
        diffuse = [
            (current[0] + 1, current[1]),  # 下
            (current[0] - 1, current[1]),  # 上
            (current[0], current[1] + 1),  # 右
            (current[0], current[1] - 1),  # 左
        ]
        if current in 此路不通:
            for e in 此路不通[current]:
                if e in diffuse:
                    diffuse.remove(e)
            # print(current, diffuse)

        for neighbor in diffuse:
            tentative_g_score = g_score[current] + 1
            if (
                0 <= neighbor[0] < len(grid)
                and 0 <= neighbor[1] < len(grid[0])
                and grid[neighbor[0]][neighbor[1]] == 0  # 0通行，1障碍
            ):
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def get_move_direction(current_node, next_node):
    main_1, cross_1 = current_node
    main_2, cross_2 = next_node

    if main_2 > main_1:
        return "下"
    elif main_2 < main_1:
        return "上"
    elif cross_2 > cross_1:
        return "右"
    else:
        return "左"


if __name__ == "__main__":
    grid = [
        [0, 0, 0], 
        [0, 0, 0], 
        [0, 0, 0]]
    start = (0, 0)
    goal = (0, 2)
    此路不通 = {
        # (0, 1): [(0, 2),(1, 1)],
        # (1,1): [ (1,2) ]
        }
    path = a_star_search(start, goal, grid, 此路不通)
    print(path)  # 输出找到的路径

    # current_node = None
    # for p in path:
    #   if not current_node:
    #     current_node = p
    #     continue

    #   pos = get_move_direction(current_node, p)
    #   print(f'{current_node}到{p}，{pos}')
    #   current_node = p
