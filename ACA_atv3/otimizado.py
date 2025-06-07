import numpy as np
import pandas as pd
import time
import warnings

# Ignorar avisos de overflow que podem ocorrer na função de boltzmann.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# DEFINIÇÃO DO AMBIENTE (MATRIZES DE TRANSIÇÃO)
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

T_matrices = [T_up, T_down, T_left, T_right]
actions_names = ["UP", "DW", "LF", "RG"]

# ==============================================================================
# FUNÇÕES AUXILIARES DO Q-LEARNING
# ==============================================================================


def calc_action_result(state, transition_state):
    cand_states = np.where(transition_state != 0)[0]
    prod_cand_states = transition_state[cand_states]
    roleta = np.cumsum(prod_cand_states)
    r = np.random.uniform()
    ind = np.where(roleta > r)[0]
    return cand_states[ind[0]]


def q_update(state, action, next_state, rw, q_matrix, alpha, gamma):
    estimate_q = rw[state] + gamma * np.max(q_matrix[next_state, :])
    q_value = q_matrix[state, action] + alpha * (estimate_q - q_matrix[state, action])
    return q_value


def choose_best_action(q_matrix, state):
    return np.argmax(q_matrix[state])


def print_policy(q_matrix, actions):
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
# FUNÇÕES PARA ESTRATÉGIAS DE EXPLORAÇÃO
# ==============================================================================


def choose_action_strategy(q_matrix, state, strategy, epsilon=0.1, tau=0.2):
    if strategy == "eps-greedy":
        if np.random.uniform() < epsilon:
            return np.random.choice([0, 1, 2, 3])
        else:
            return choose_best_action(q_matrix, state)

    elif strategy == "boltzmann":
        q_values = q_matrix[state, :]
        q_values_stable = q_values - np.max(q_values)
        probabilities = np.exp(q_values_stable / tau) / np.sum(
            np.exp(q_values_stable / tau)
        )
        # Lidando com NaN que pode ocorrer se tau for muito pequeno
        if np.isnan(probabilities).any():
            return choose_best_action(q_matrix, state)
        return np.random.choice([0, 1, 2, 3], p=probabilities)

    else:  # 'random'
        return np.random.choice([0, 1, 2, 3])


# ==============================================================================
# FUNÇÕES DE EXECUÇÃO E SIMULAÇÃO
# ==============================================================================


def run_q_learning(
    alpha, gamma, trajectories, exploration_strategy="random", epsilon=0.1, tau=0.2
):
    q_matrix = np.zeros((11, 4))
    q_matrix[9, :] = -1
    q_matrix[10, :] = 1
    rw = np.full(11, -0.04)
    rw[9] = -1
    rw[10] = 1

    for _ in range(trajectories):
        state = 0
        is_terminal = False
        while not is_terminal:
            action_trial = choose_action_strategy(
                q_matrix, state, exploration_strategy, epsilon, tau
            )
            transition_state = T_matrices[action_trial][state, :]
            next_state = calc_action_result(state, transition_state)
            q_matrix[state, action_trial] = q_update(
                state, action_trial, next_state, rw, q_matrix, alpha, gamma
            )
            state = next_state
            if state == 9 or state == 10:
                is_terminal = True
    return q_matrix


def simulate_policy(q_matrix, num_simulations):
    total_rewards = []
    rw = np.full(11, -0.04)
    rw[9] = -1
    rw[10] = 1

    for _ in range(num_simulations):
        r_total = 0
        state = 0
        for _ in range(100):
            action_trial = choose_best_action(q_matrix, state)
            transition_state = T_matrices[action_trial][state]
            next_state = calc_action_result(state, transition_state)
            r_total += rw[next_state]
            state = next_state
            if state == 9 or state == 10:
                break
        total_rewards.append(r_total)
    return np.mean(total_rewards)


# ==============================================================================
# SCRIPT PRINCIPAL PARA EXECUÇÃO DOS ITENS
# ==============================================================================

TRAJECTORIES = 30
NUM_TRAININGS = 10
NUM_SIMULATIONS = 100

# --- ITEM 1: Avaliação de alpha e gamma ---
print("=" * 60)
print("INICIANDO ITEM 1: Otimização de alpha e gamma")
print("=" * 60)
alpha_options = [0.1, 0.3, 0.5, 0.7, 0.9]
gamma_options = [0.1, 0.3, 0.5, 0.7, 0.9]
results_item1 = pd.DataFrame(index=alpha_options, columns=gamma_options, dtype=float)

for alpha in alpha_options:
    for gamma in gamma_options:
        avg_rewards = [
            simulate_policy(run_q_learning(alpha, gamma, TRAJECTORIES), NUM_SIMULATIONS)
            for _ in range(NUM_TRAININGS)
        ]
        results_item1.loc[alpha, gamma] = np.mean(avg_rewards)

print("\n--- Resultados Finais (Item 1) ---")
print(results_item1)
best_alpha = results_item1.stack().idxmax()[0]
best_gamma = results_item1.stack().idxmax()[1]
print(f"\nMelhor configuração encontrada: alpha={best_alpha}, gamma={best_gamma}")

# --- ITEM 1.5: Otimização dos Parâmetros de Exploração ---
print("\n\n" + "=" * 60)
print("INICIANDO ITEM 1.5: Otimização de Epsilon (eps-greedy)")
print(f"Usando alpha={best_alpha} e gamma={best_gamma}")
print("=" * 60)
epsilon_options = [0.05, 0.1, 0.2, 0.3, 0.5]
eps_results = {}
for eps in epsilon_options:
    avg_rewards = [
        simulate_policy(
            run_q_learning(
                best_alpha, best_gamma, TRAJECTORIES, "eps-greedy", epsilon=eps
            ),
            NUM_SIMULATIONS,
        )
        for _ in range(NUM_TRAININGS)
    ]
    eps_results[eps] = np.mean(avg_rewards)
    print(f"Epsilon={eps:<4} -> Recompensa Média: {eps_results[eps]:.4f}")

best_epsilon = max(eps_results, key=eps_results.get)
print(f"\nMelhor Epsilon encontrado: {best_epsilon}")

print("\n\n" + "=" * 60)
print("INICIANDO ITEM 1.5: Otimização de Tau (Boltzmann)")
print(f"Usando alpha={best_alpha} e gamma={best_gamma}")
print("=" * 60)
tau_options = [0.01, 0.05, 0.1, 0.2, 0.5]
tau_results = {}
for tau in tau_options:
    avg_rewards = [
        simulate_policy(
            run_q_learning(best_alpha, best_gamma, TRAJECTORIES, "boltzmann", tau=tau),
            NUM_SIMULATIONS,
        )
        for _ in range(NUM_TRAININGS)
    ]
    tau_results[tau] = np.mean(avg_rewards)
    print(f"Tau={tau:<4} -> Recompensa Média: {tau_results[tau]:.4f}")

best_tau = max(tau_results, key=tau_results.get)
print(f"\nMelhor Tau encontrado: {best_tau}")


# --- ITEM 2: Comparação Final das Estratégias de Exploração ---
print("\n\n" + "=" * 60)
print("INICIANDO ITEM 2: Comparação Final das Estratégias")
print(
    f"Usando alpha={best_alpha}, gamma={best_gamma}, epsilon={best_epsilon}, tau={best_tau}"
)
print("=" * 60)

strategies_to_compare = {
    "random": {},
    "eps-greedy": {"epsilon": best_epsilon},
    "boltzmann": {"tau": best_tau},
}
results_item2 = {}

for name, params in strategies_to_compare.items():
    print(f"Avaliando estratégia: {name} com parâmetros {params}...")
    avg_rewards = [
        simulate_policy(
            run_q_learning(best_alpha, best_gamma, TRAJECTORIES, name, **params),
            NUM_SIMULATIONS,
        )
        for _ in range(NUM_TRAININGS)
    ]
    results_item2[name] = np.mean(avg_rewards)

print("\n--- Resultados Finais (Item 2) ---")
results_df_item2 = pd.DataFrame.from_dict(
    results_item2, orient="index", columns=["Recompensa Média"]
)
print(results_df_item2.sort_values(by="Recompensa Média", ascending=False))

best_strategy_name = results_df_item2["Recompensa Média"].idxmax()
best_strategy_params = strategies_to_compare[best_strategy_name]

print(f"\nMelhor estratégia geral encontrada: {best_strategy_name}")
print("Gerando uma política final com a melhor configuração...")
final_q_matrix = run_q_learning(
    best_alpha, best_gamma, TRAJECTORIES, best_strategy_name, **best_strategy_params
)
print_policy(final_q_matrix, actions_names)
