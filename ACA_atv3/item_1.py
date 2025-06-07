import numpy as np
import pandas as pd
import time

# ==============================================================================
# CEL 1: DEFINIÇÃO DO AMBIENTE (MATRIZES DE TRANSIÇÃO)
# ==============================================================================
# Matriz de transição para a ação UP
T_up = np.zeros((11, 11))
T_up[0, 3] = 0.1
T_up[0, 0] = 0.1
T_up[0, 1] = 0.8
T_up[1, 1] = 0.2
T_up[1, 2] = 0.8
T_up[2, 4] = 0.1
T_up[2, 2] = 0.9
T_up[3, 5] = 0.1
T_up[3, 3] = 0.8
T_up[3, 0] = 0.1
T_up[4, 7] = 0.1
T_up[4, 2] = 0.1
T_up[4, 4] = 0.8
T_up[5, 8] = 0.1
T_up[5, 3] = 0.1
T_up[5, 6] = 0.8
T_up[6, 9] = 0.1
T_up[6, 6] = 0.1
T_up[6, 7] = 0.8
T_up[7, 10] = 0.1
T_up[7, 4] = 0.1
T_up[7, 7] = 0.8
T_up[8, 8] = 0.1
T_up[8, 5] = 0.1
T_up[8, 9] = 0.8

# Matriz de transição para a ação DOWN
T_down = np.zeros((11, 11))
T_down[0, 0] = 0.9
T_down[0, 3] = 0.1
T_down[1, 0] = 0.8
T_down[1, 1] = 0.2
T_down[2, 1] = 0.8
T_down[2, 2] = 0.1
T_down[2, 4] = 0.1
T_down[3, 0] = 0.1
T_down[3, 5] = 0.1
T_down[3, 3] = 0.8
T_down[4, 4] = 0.8
T_down[4, 2] = 0.1
T_down[4, 7] = 0.1
T_down[5, 5] = 0.8
T_down[5, 3] = 0.1
T_down[5, 8] = 0.1
T_down[6, 5] = 0.8
T_down[6, 6] = 0.1
T_down[6, 9] = 0.1
T_down[7, 6] = 0.8
T_down[7, 4] = 0.1
T_down[7, 10] = 0.1
T_down[8, 8] = 0.9
T_down[8, 5] = 0.1

# Matriz de transição para a ação LEFT
T_left = np.zeros((11, 11))
T_left[0, 0] = 0.9
T_left[0, 1] = 0.1
T_left[1, 0] = 0.1
T_left[1, 2] = 0.1
T_left[1, 1] = 0.8
T_left[2, 2] = 0.9
T_left[2, 1] = 0.1
T_left[3, 0] = 0.8
T_left[3, 3] = 0.2
T_left[4, 2] = 0.8
T_left[4, 4] = 0.2
T_left[5, 3] = 0.8
T_left[5, 5] = 0.1
T_left[5, 6] = 0.1
T_left[6, 6] = 0.8
T_left[6, 5] = 0.1
T_left[6, 7] = 0.1
T_left[7, 4] = 0.8
T_left[7, 7] = 0.1
T_left[7, 6] = 0.1
T_left[8, 5] = 0.8
T_left[8, 8] = 0.1
T_left[8, 9] = 0.1

# Matriz de transição para a ação RIGHT
T_right = np.zeros((11, 11))
T_right[0, 3] = 0.8
T_right[0, 0] = 0.1
T_right[0, 1] = 0.1
T_right[1, 1] = 0.8
T_right[1, 0] = 0.1
T_right[1, 2] = 0.1
T_right[2, 4] = 0.8
T_right[2, 2] = 0.1
T_right[2, 1] = 0.1
T_right[3, 5] = 0.8
T_right[3, 3] = 0.2
T_right[4, 7] = 0.8
T_right[4, 4] = 0.2
T_right[5, 8] = 0.9
T_right[5, 5] = 0.1
T_right[5, 6] = 0.1
T_right[6, 9] = 0.8
T_right[6, 5] = 0.1
T_right[6, 7] = 0.1
T_right[7, 10] = 0.8
T_right[7, 6] = 0.1
T_right[7, 7] = 0.1
T_right[8, 9] = 0.1
T_right[8, 8] = 0.9

# Agrupando as matrizes de transição
T_matrices = [T_up, T_down, T_left, T_right]
actions_names = ["UP", "DW", "LF", "RG"]

# ==============================================================================
# CEL 2-6: FUNÇÕES AUXILIARES DO Q-LEARNING
# ==============================================================================


def calc_action_result(state, transition_state):
    """Simula o resultado estocástico de aplicar uma ação."""
    cand_states = np.where(transition_state != 0)[0]
    prod_cand_states = transition_state[cand_states]
    roleta = np.cumsum(prod_cand_states)
    r = np.random.uniform()
    ind = np.where(roleta > r)[0]
    return cand_states[ind[0]]


def q_update(state, action, next_state, rw, q_matrix, alpha, gamma):
    """Atualiza o valor Q para um par (estado, ação)."""
    estimate_q = rw[state] + gamma * np.max(q_matrix[next_state, :])
    q_value = q_matrix[state, action] + alpha * (estimate_q - q_matrix[state, action])
    return q_value


def choose_best_action(q_matrix, state):
    """Retorna a melhor ação para um estado com base na matriz Q."""
    return np.argmax(q_matrix[state])


def print_policy(q_matrix, actions):
    """Imprime a política formatada."""
    policy = np.argmax(q_matrix, axis=1)
    s1 = " ".join([actions[policy[2]], actions[policy[4]], actions[policy[7]], "+1"])
    s2 = " ".join([actions[policy[1]], "*", actions[policy[6]], "-1"])
    s3 = " ".join(
        [actions[policy[0]], actions[policy[3]], actions[policy[5]], actions[policy[8]]]
    )
    print("\n--- Política Aprendida ---")
    print(s1, "\n", s2, "\n", s3)
    print("------------------------\n")


# ==============================================================================
# FUNÇÕES PARA O ITEM 2: ESTRATÉGIAS DE EXPLORAÇÃO
# ==============================================================================


def choose_action_strategy(q_matrix, state, strategy, epsilon=0.1, tau=0.2):
    """
    Escolhe uma ação com base na estratégia de exploração especificada.

    Args:
        q_matrix (np.array): A matriz Q atual.
        state (int): O estado atual.
        strategy (str): 'random', 'eps-greedy', ou 'boltzmann'.
        epsilon (float): Parâmetro para Epsilon-Greedy.
        tau (float): Parâmetro de temperatura para Boltzmann.

    Returns:
        int: O índice da ação escolhida.
    """
    if strategy == "eps-greedy":
        if np.random.uniform() < epsilon:
            return np.random.choice([0, 1, 2, 3])  # Explorar
        else:
            return choose_best_action(q_matrix, state)  # Explotar

    elif strategy == "boltzmann":
        q_values = q_matrix[state, :]
        # Adicionar verificação para evitar overflow com exp
        q_values_stable = q_values - np.max(q_values)
        probabilities = np.exp(q_values_stable / tau) / np.sum(
            np.exp(q_values_stable / tau)
        )
        return np.random.choice([0, 1, 2, 3], p=probabilities)

    else:  # 'random' (padrão)
        return np.random.choice([0, 1, 2, 3])


# ==============================================================================
# FUNÇÕES DE EXECUÇÃO E SIMULAÇÃO
# ==============================================================================


def run_q_learning(alpha, gamma, trajectories, exploration_strategy="random"):
    """
    Executa um ciclo completo de treinamento do Q-Learning.

    Returns:
        np.array: A matriz Q treinada.
    """
    # Inicialização da matriz Q
    q_matrix = np.zeros((11, 4))
    q_matrix[9, :] = -1  # Estado terminal -1
    q_matrix[10, :] = 1  # Estado terminal +1

    # Vetor de recompensas
    rw = np.full(11, -0.04)
    rw[9] = -1
    rw[10] = 1

    for _ in range(trajectories):
        state = 0  # Estado inicial
        is_terminal = False
        while not is_terminal:
            # Escolher ação com base na estratégia
            action_trial = choose_action_strategy(q_matrix, state, exploration_strategy)

            # Obter próximo estado
            transition_state = T_matrices[action_trial][state, :]
            next_state = calc_action_result(state, transition_state)

            # Atualizar Q-value
            q_matrix[state, action_trial] = q_update(
                state, action_trial, next_state, rw, q_matrix, alpha, gamma
            )

            state = next_state

            if state == 9 or state == 10:
                is_terminal = True

    return q_matrix


def simulate_policy(q_matrix, num_simulations):
    """
    Simula a execução da política aprendida e calcula a recompensa média.
    """
    total_rewards = []

    # Vetor de recompensas (necessário para a simulação)
    rw = np.full(11, -0.04)
    rw[9] = -1
    rw[10] = 1

    for _ in range(num_simulations):
        r_total = 0
        state = 0  # Estado inicial
        is_terminal = False

        # Limitar o número de passos para evitar loops infinitos em políticas ruins
        for _ in range(100):
            action_trial = choose_best_action(q_matrix, state)
            transition_state = T_matrices[action_trial][state]
            next_state = calc_action_result(state, transition_state)
            r_total += rw[next_state]
            state = next_state
            if state == 9 or state == 10:
                is_terminal = True
                break

        total_rewards.append(r_total)

    return np.mean(total_rewards)


# ==============================================================================
# SCRIPT PRINCIPAL PARA EXECUÇÃO DOS ITENS
# ==============================================================================

# --- PARÂMETROS GERAIS ---
TRAJECTORIES = 30
NUM_TRAININGS = 10
NUM_SIMULATIONS = 100

# --- ITEM 1: Avaliação de alpha e gamma ---
print("=" * 50)
print("INICIANDO ITEM 1: Otimização de alpha e gamma")
print("=" * 50)

alpha_options = [0.1, 0.3, 0.5, 0.7, 0.9]
gamma_options = [0.1, 0.3, 0.5, 0.7, 0.9]

results_item1 = pd.DataFrame(index=alpha_options, columns=gamma_options, dtype=float)

start_time = time.time()
for alpha in alpha_options:
    for gamma in gamma_options:
        avg_rewards_for_config = []
        for i in range(NUM_TRAININGS):
            # 1. Treinar o modelo
            q_matrix_trained = run_q_learning(
                alpha, gamma, TRAJECTORIES, exploration_strategy="random"
            )

            # 2. Avaliar a política aprendida
            avg_reward = simulate_policy(q_matrix_trained, NUM_SIMULATIONS)
            avg_rewards_for_config.append(avg_reward)

        # 3. Calcular a média final para a configuração (alpha, gamma)
        final_avg_reward = np.mean(avg_rewards_for_config)
        results_item1.loc[alpha, gamma] = final_avg_reward
        print(
            f"Alpha={alpha}, Gamma={gamma} -> Recompensa Média: {final_avg_reward:.4f}"
        )

end_time = time.time()

print("\n--- Resultados Finais (Item 1) ---")
print(results_item1)
print(f"\nTempo de execução do Item 1: {end_time - start_time:.2f} segundos")

# Encontrar a melhor configuração
best_alpha = results_item1.stack().idxmax()[0]
best_gamma = results_item1.stack().idxmax()[1]
print(f"\nMelhor configuração encontrada: alpha={best_alpha}, gamma={best_gamma}")


# --- ITEM 2: Avaliação das Estratégias de Exploração ---
print("\n\n" + "=" * 50)
print("INICIANDO ITEM 2: Análise de Estratégias de Exploração")
print(f"Usando alpha={best_alpha} e gamma={best_gamma}")
print("=" * 50)

exploration_strategies = ["random", "eps-greedy", "boltzmann"]
results_item2 = {}

start_time_item2 = time.time()
for strategy in exploration_strategies:
    avg_rewards_for_strategy = []
    print(f"Avaliando estratégia: {strategy}...")
    for i in range(NUM_TRAININGS):
        # 1. Treinar com a nova estratégia
        q_matrix_trained = run_q_learning(
            best_alpha, best_gamma, TRAJECTORIES, exploration_strategy=strategy
        )

        # 2. Avaliar a política
        avg_reward = simulate_policy(q_matrix_trained, NUM_SIMULATIONS)
        avg_rewards_for_strategy.append(avg_reward)

    # 3. Calcular a média final para a estratégia
    final_avg_reward = np.mean(avg_rewards_for_strategy)
    results_item2[strategy] = final_avg_reward

end_time_item2 = time.time()

print("\n--- Resultados Finais (Item 2) ---")
results_df_item2 = pd.DataFrame.from_dict(
    results_item2, orient="index", columns=["Recompensa Média"]
)
print(results_df_item2)

# Imprimir a política final com a melhor estratégia
best_strategy = results_df_item2["Recompensa Média"].idxmax()
print(f"\nMelhor estratégia encontrada: {best_strategy}")
print(
    "Gerando uma política final com a melhor configuração (alpha, gamma, strategy)..."
)
final_q_matrix = run_q_learning(
    best_alpha, best_gamma, TRAJECTORIES, exploration_strategy=best_strategy
)
print_policy(final_q_matrix, actions_names)
print(
    f"\nTempo de execução do Item 2: {end_time_item2 - start_time_item2:.2f} segundos"
)
