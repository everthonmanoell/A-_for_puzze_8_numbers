import heapq
import copy

# ---------- DEFINIÇÕES ---------- #

goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

goal_positions = {goal_state[i][j]: (i, j) for i in range(3) for j in range(3)}


def print_state(state):
    return "\n".join(
        [" ".join([str(num) if num != 0 else " " for num in row]) for row in state]
    )


def is_goal(state):
    return state == goal_state


def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j


def manhattan_distance(state):
    dist = 0
    for i in range(3):
        for j in range(3):
            num = state[i][j]
            if num != 0:
                goal_i, goal_j = goal_positions[num]
                dist += abs(i - goal_i) + abs(j - goal_j)
    return dist


def get_neighbors(state):
    neighbors = []
    i, j = find_blank(state)
    directions = {
        "Cima": (-1, 0),
        "Baixo": (1, 0),
        "Esquerda": (0, -1),
        "Direita": (0, 1),
    }
    for action, (di, dj) in directions.items():
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            new_state = copy.deepcopy(state)
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            neighbors.append((new_state, action))
    return neighbors


def a_star(start_state):
    open_list = []
    closed_set = set()

    h = manhattan_distance(start_state)
    start_node = (h, 0, start_state, [], None)
    heapq.heappush(open_list, start_node)

    step = 0

    while open_list:
        f, g, current_state, path, action = heapq.heappop(open_list)
        state_tuple = tuple(tuple(row) for row in current_state)

        if state_tuple in closed_set:
            continue
        closed_set.add(state_tuple)

        log_step(step, current_state, f, g, path, action)
        step += 1

        if is_goal(current_state):
            print("\n*** Solução encontrada! ***")
            return path + [current_state], g

        for neighbor, act in get_neighbors(current_state):
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            if neighbor_tuple in closed_set:
                continue
            new_g = g + 1
            new_h = manhattan_distance(neighbor)
            new_f = new_g + new_h
            new_path = path + [current_state]
            heapq.heappush(open_list, (new_f, new_g, neighbor, new_path, act))

    print("Falha: Não foi possível encontrar uma solução.")
    return None, None


def log_step(step, state, f, g, path, action):
    print(f"\nPasso {step}:")
    print("Estado atual com f(n) = g + h = {} + {} = {}".format(g, f - g, f))
    print(print_state(state))
    print("Ação que levou a este estado:", action)
    print("Tamanho do caminho até aqui:", len(path))


def print_solution(path, cost):
    print("\nCAMINHO FINAL (do início ao objetivo):")
    for i, estado in enumerate(path):
        print(f"\nEstado {i}:")
        print(print_state(estado))
    print("\nCusto total da solução:", cost)


def main():
    initial_state = [[2, 8, 3], [1, 6, 4], [7, 0, 5]]

    solucao, custo_total = a_star(initial_state)

    if solucao:
        print_solution(solucao, custo_total)


if __name__ == "__main__":
    main()
