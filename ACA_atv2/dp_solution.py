import heapq
from copy import deepcopy


class Estado:
    def __init__(self, tabuleiro, movimento_anterior=None, custo_g=0):
        self.tabuleiro = tabuleiro
        self.movimento_anterior = movimento_anterior
        self.custo_g = custo_g
        self.custo_h = self.calcular_heuristica()
        self.custo_f = self.custo_g + self.custo_h

    def __lt__(self, outro):
        return self.custo_f < outro.custo_f

    def __eq__(self, outro):
        return self.tabuleiro == outro.tabuleiro

    def __hash__(self):
        return hash(tuple(tuple(linha) for linha in self.tabuleiro))

    def calcular_heuristica(self):
        """Calcula a distância de Manhattan para cada peça"""
        h = 0
        objetivo = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        for i in range(3):
            for j in range(3):
                valor = self.tabuleiro[i][j]
                if valor != 0:  # Ignora o espaço vazio
                    linha_objetivo, coluna_objetivo = self.encontrar_posicao_objetivo(
                        valor, objetivo
                    )
                    h += abs(i - linha_objetivo) + abs(j - coluna_objetivo)
        return h

    def encontrar_posicao_objetivo(self, valor, objetivo):
        """Encontra a posição correta de uma peça no estado objetivo"""
        for i in range(3):
            for j in range(3):
                if objetivo[i][j] == valor:
                    return i, j
        return None, None

    def encontrar_espaco_vazio(self):
        """Encontra a posição do espaço vazio (0)"""
        for i in range(3):
            for j in range(3):
                if self.tabuleiro[i][j] == 0:
                    return i, j
        return None, None

    def gerar_filhos(self):
        """Gera todos os estados filhos possíveis"""
        filhos = []
        i, j = self.encontrar_espaco_vazio()

        # Movimentos possíveis: cima, baixo, esquerda, direita
        movimentos = []
        if i > 0:
            movimentos.append((-1, 0, "↑"))  # Cima
        if i < 2:
            movimentos.append((1, 0, "↓"))  # Baixo
        if j > 0:
            movimentos.append((0, -1, "←"))  # Esquerda
        if j < 2:
            movimentos.append((0, 1, "→"))  # Direita

        for di, dj, movimento in movimentos:
            novo_tabuleiro = deepcopy(self.tabuleiro)
            # Troca o espaço vazio com a peça adjacente
            novo_tabuleiro[i][j], novo_tabuleiro[i + di][j + dj] = (
                novo_tabuleiro[i + di][j + dj],
                novo_tabuleiro[i][j],
            )
            filhos.append(Estado(novo_tabuleiro, movimento, self.custo_g + 1))

        return filhos

    def eh_objetivo(self):
        """Verifica se o estado é o objetivo"""
        objetivo = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        return self.tabuleiro == objetivo

    def caminho_desde_inicio(self):
        """Recupera o caminho desde o estado inicial"""
        caminho = []
        estado = self
        while estado.movimento_anterior is not None:
            caminho.append((estado.movimento_anterior, estado.tabuleiro))
            # Precisamos recriar o estado anterior para continuar o caminho
            # Isso é simplificado aqui - numa implementação real deveríamos ter referência ao pai
            break  # Esta parte precisa ser reimplementada corretamente
        caminho.reverse()
        return caminho

    def __str__(self):
        return (
            "\n".join([" ".join(map(str, linha)) for linha in self.tabuleiro])
            + f"\nf(n) = {self.custo_f} (g={self.custo_g}, h={self.custo_h})\n"
        )


def a_estrela(estado_inicial):
    fronteira = []
    heapq.heappush(fronteira, estado_inicial)
    explorados = set()

    iteracao = 0

    while fronteira:
        print(f"\n--- Iteração {iteracao} ---")
        print("Fronteira atual:")
        for i, estado in enumerate(
            fronteira[:5]
        ):  # Mostra apenas os 5 primeiros para não poluir
            print(f"Estado {i}:")
            print(estado)
        if len(fronteira) > 5:
            print(f"... mais {len(fronteira)-5} estados na fronteira ...")

        estado_atual = heapq.heappop(fronteira)

        print(f"\nExpandindo estado com f(n) = {estado_atual.custo_f}:")
        print(estado_atual)

        if estado_atual.eh_objetivo():
            print("Objetivo alcançado!")
            return estado_atual

        # Usamos apenas o tabuleiro para comparação
        tabuleiro_tuple = tuple(tuple(linha) for linha in estado_atual.tabuleiro)
        if tabuleiro_tuple in explorados:
            continue

        explorados.add(tabuleiro_tuple)

        for filho in estado_atual.gerar_filhos():
            filho_tuple = tuple(tuple(linha) for linha in filho.tabuleiro)
            if filho_tuple not in explorados:
                # Verifica se já está na fronteira com custo maior
                encontrado = False
                for i, est in enumerate(fronteira):
                    if est.tabuleiro == filho.tabuleiro:
                        if filho.custo_f < est.custo_f:
                            fronteira[i] = filho
                            heapq.heapify(fronteira)
                        encontrado = True
                        break
                if not encontrado:
                    heapq.heappush(fronteira, filho)

        iteracao += 1
        if iteracao > 1000:  # Limite de segurança
            print("Limite de iterações alcançado!")
            break

    return None


# Estado inicial do exemplo
tabuleiro_inicial = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]

estado_inicial = Estado(tabuleiro_inicial)
solucao = a_estrela(estado_inicial)

if solucao:
    print("\n=== Solução encontrada ===")
    print(f"Custo total: {solucao.custo_g} movimentos")
    print("\nPasso a passo:")
    # Implementação simplificada do caminho - precisa ser melhorada
    print("Estado inicial:")
    print(estado_inicial)
    print("\nEstado final:")
    print(solucao)
else:
    print("Não foi encontrada solução.")
