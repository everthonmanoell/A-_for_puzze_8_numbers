# 1 - imports
import heapq  # biblioteca para manipular filas de prioridade - usada para selecionar o estado com menor custo f(n)
from copy import (
    deepcopy,
)  # usado para fazer cópias profundas dos tabuleiros, evitando modificar os originais


# 2 - Classe estado
class Estado:
    # Objetivo da roda - estado final desejado
    objetivo = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    # Construtor
    def __init__(self, tabuleiro, movimento_anterior=None, custo_g=0, pai=None):
        self.tabuleiro = tabuleiro  # matriz 3x3 representando o estado atual
        self.movimento_anterior = movimento_anterior  # último movimento feito (usado para reconstruir caminho)
        self.custo_g = custo_g  # custo real acumulado do caminho (g)
        self.pai = (
            pai  # referência ao estado anterior (pai), para reconstrução do caminho
        )
        self.custo_h = (
            self.calcular_heuristica()
        )  # cálculo da heurística (distância de Manhattan)
        self.custo_f = (
            self.custo_g + self.custo_h
        )  # função de avaliação A* -> f(n) = g(n) + h(n)

    def __lt__(
        self, outro
    ):  # comparação de estados para heapq - define prioridade com base no menor custo f(n)
        return self.custo_f < outro.custo_f

    def __eq__(
        self, outro
    ):  # comparação entre estados - necessário para verificar se já foi visitado
        return self.tabuleiro == outro.tabuleiro

    def __hash__(
        self,
    ):  # gera um hash único para cada tabuleiro, para armazenar em conjuntos (set)
        return hash(
            tuple(tuple(linha) for linha in self.tabuleiro)
        )  # transforma a matriz em tupla de tuplas para ser hasheável

    # Função heurística de Manhattan
    def calcular_heuristica(self):
        h = 0  # acumulador da heurística
        for i in range(3):
            for j in range(3):
                valor = self.tabuleiro[i][j]
                if valor != 0:  # ignorar o espaço vazio (0)
                    linha_objetivo, coluna_objetivo = self.encontrar_posicao_objetivo(
                        valor
                    )
                    h += abs(i - linha_objetivo) + abs(
                        j - coluna_objetivo
                    )  # soma das distâncias em linha e coluna
        return h

    # Retorna a posição (linha, coluna) do valor no estado objetivo
    def encontrar_posicao_objetivo(self, valor):
        for i in range(3):
            for j in range(3):
                if Estado.objetivo[i][j] == valor:
                    return i, j
        return None, None  # segurança - valor não encontrado

    # Encontra a posição atual do espaço vazio (valor 0)
    def encontrar_espaco_vazio(self):
        for i in range(3):
            for j in range(3):
                if self.tabuleiro[i][j] == 0:
                    return i, j
        return None, None  # segurança - espaço vazio não encontrado

    # Geração de novos estados (filhos) a partir do estado atual
    def gerar_filhos(self):
        filhos = []  # lista para armazenar os filhos
        i, j = self.encontrar_espaco_vazio()  # encontra a posição do 0 (espaço vazio)

        movimentos = []  # lista de movimentos possíveis
        if i > 0:
            movimentos.append((-1, 0, "↑"))  # mover para cima
        if i < 2:
            movimentos.append((1, 0, "↓"))  # mover para baixo
        if j > 0:
            movimentos.append((0, -1, "←"))  # mover para a esquerda
        if j < 2:
            movimentos.append((0, 1, "→"))  # mover para a direita

        # Geração dos novos estados com base nos movimentos válidos
        for di, dj, movimento in movimentos:
            novo_tabuleiro = deepcopy(self.tabuleiro)  # cópia profunda do tabuleiro
            # troca o espaço vazio com o número vizinho
            novo_tabuleiro[i][j], novo_tabuleiro[i + di][j + dj] = (
                novo_tabuleiro[i + di][j + dj],
                novo_tabuleiro[i][j],
            )
            # cria o novo estado com o movimento registrado como tupla (numero movido, direção)
            filhos.append(
                Estado(
                    novo_tabuleiro,
                    (
                        self.tabuleiro[i + di][j + dj],
                        movimento,
                    ),  # número que foi movido e direção
                    self.custo_g + 1,  # incremento no custo real
                    self,  # referência ao pai (estado atual)
                )
            )

        return filhos  # retorna lista de estados filhos

    # Verifica se o estado atual é o estado objetivo
    def eh_objetivo(self):
        return self.tabuleiro == Estado.objetivo

    # Reconstrói o caminho do estado inicial até o estado atual
    def caminho_desde_inicio(self):
        caminho = []  # lista para armazenar o caminho
        estado = self
        while estado.pai is not None:  # percorre os pais até o início
            caminho.append(
                (estado.movimento_anterior, estado.tabuleiro)
            )  # adiciona tupla (movimento, tabuleiro)
            estado = estado.pai
        caminho.reverse()  # inverte para ficar do início ao fim
        return caminho

    # Representação em string do estado (tabuleiro + função f)
    def __str__(self):
        return (
            "\n".join(
                [" ".join(map(str, linha)) for linha in self.tabuleiro]
            )  # imprime o tabuleiro linha por linha
            + f"\nf(n) = {self.custo_f} (g={self.custo_g}, h={self.custo_h})\n"  # mostra os custos
        )


# 3 - A* (função principal do algoritmo A*)
def a_estrela(estado_inicial):
    fronteira = []  # fila de prioridade com os estados a serem explorados
    heapq.heappush(fronteira, estado_inicial)  # insere o estado inicial na fila
    explorados = set()  # conjunto de estados já explorados

    iteracao = 0  # contador de iterações

    while fronteira:
        print(f"\n--- Iteração {iteracao} ---")  # exibe a iteração atual
        print("Fronteira atual:")
        for i, estado in enumerate(
            fronteira[:5]
        ):  # imprime os primeiros estados da fronteira (no máximo 5)
            print(f"Estado {i}:")
            print(estado)
        if len(fronteira) > 5:
            print(
                f"... mais {len(fronteira)-5} estados na fronteira ..."
            )  # aviso de estados adicionais

        estado_atual = heapq.heappop(fronteira)  # seleciona o estado com menor f(n)

        print(f"\nExpandindo estado com f(n) = {estado_atual.custo_f}:")
        print(estado_atual)

        if estado_atual.eh_objetivo():  # verifica se é o objetivo
            print("Objetivo alcançado!")
            return estado_atual  # retorna o estado final

        tabuleiro_tuple = tuple(
            tuple(linha) for linha in estado_atual.tabuleiro
        )  # transforma em tupla para usar no set
        if tabuleiro_tuple in explorados:  # ignora se já foi explorado
            continue

        explorados.add(tabuleiro_tuple)  # marca como explorado

        for filho in estado_atual.gerar_filhos():  # gera os filhos do estado atual
            filho_tuple = tuple(tuple(linha) for linha in filho.tabuleiro)
            if filho_tuple not in explorados:  # se ainda não foi explorado
                encontrado = False  # flag para indicar se já está na fronteira
                for i, est in enumerate(
                    fronteira
                ):  # verifica se o filho já está na fronteira
                    if est.tabuleiro == filho.tabuleiro:
                        if (
                            filho.custo_f < est.custo_f
                        ):  # se o novo caminho é melhor, substitui
                            fronteira[i] = filho
                            heapq.heapify(fronteira)  # reorganiza a fila
                        encontrado = True
                        break
                if not encontrado:  # se não estava na fronteira, adiciona
                    heapq.heappush(fronteira, filho)

        iteracao += 1  # próxima iteração
        if iteracao > 1000:  # limite de segurança para evitar loops infinitos
            print("Limite de iterações alcançado!")
            break

    return None  # caso não encontre solução


# 4 - Execução
if __name__ == "__main__":
    tabuleiro_inicial = [
        [1, 8, 2],
        [0, 4, 3],
        [7, 6, 5],
    ]  # configuração inicial do tabuleiro
    estado_inicial = Estado(tabuleiro_inicial)  # cria o estado inicial
    solucao = a_estrela(estado_inicial)  # executa o algoritmo A*

    if solucao:
        print("\n=== Solução encontrada ===")
        print(
            f"Custo total: {solucao.custo_g} movimentos"
        )  # mostra o custo final da solução
        print("\nPasso a passo:")
        print("Estado inicial:")
        print(
            "\n".join(" ".join(map(str, linha)) for linha in tabuleiro_inicial)
        )  # imprime o tabuleiro inicial

        for (
            movimento,
            tabuleiro,
        ) in solucao.caminho_desde_inicio():  # percorre os movimentos da solução
            num, dir = movimento  # desempacota a tupla (número movido, direção)
            print(f"\nMovimento: {num} {dir}")
            print(
                "\n".join(" ".join(map(str, linha)) for linha in tabuleiro)
            )  # imprime o tabuleiro após o movimento

        print("\nEstado final:")
        print(solucao)  # imprime o estado objetivo alcançado
    else:
        print("Não foi encontrada solução.")  # caso não encontre caminho
