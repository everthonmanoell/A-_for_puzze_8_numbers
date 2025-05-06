import heapq
from copy import deepcopy


class Estado:
    def __init__(self, tabuleiro, movimento_anterior=None, custo_g=0, pai=None):
        self.tabuleiro = tabuleiro
        self.movimento_anterior = movimento_anterior  # (numero_movido, direcao)
        self.custo_g = custo_g
        self.pai = pai
        self.custo_h = self.calcular_heuristica()
        self.custo_f = self.custo_g + self.custo_h

    def __lt__(self, outro):
        return self.custo_f < outro.custo_f

    def __eq__(self, outro):
        return self.tabuleiro == outro.tabuleiro

    def __hash__(self):
        return hash(tuple(tuple(linha) for linha in self.tabuleiro))

    def calcular_heuristica(self):
        h = 0
        objetivo = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        for i in range(3):
            for j in range(3):
                valor = self.tabuleiro[i][j]
                if valor != 0:
                    linha_obj, col_obj = self.encontrar_posicao_objetivo(
                        valor, objetivo
                    )
                    h += abs(i - linha_obj) + abs(j - col_obj)
        return h

    def encontrar_posicao_objetivo(self, valor, objetivo):
        for i in range(3):
            for j in range(3):
                if objetivo[i][j] == valor:
                    return i, j
        return None, None

    def encontrar_espaco_vazio(self):
        for i in range(3):
            for j in range(3):
                if self.tabuleiro[i][j] == 0:
                    return i, j
        return None, None

    def gerar_filhos(self):
        filhos = []
        i, j = self.encontrar_espaco_vazio()
        movimentos = []
        if i > 0:
            movimentos.append((-1, 0, "↓"))  # Peça acima desce
        if i < 2:
            movimentos.append((1, 0, "↑"))  # Peça abaixo sobe
        if j > 0:
            movimentos.append((0, -1, "→"))  # Peça à esquerda vai para direita
        if j < 2:
            movimentos.append((0, 1, "←"))  # Peça à direita vai para esquerda

        for di, dj, direcao in movimentos:
            novo_tabuleiro = deepcopy(self.tabuleiro)
            numero_movido = novo_tabuleiro[i + di][j + dj]
            novo_tabuleiro[i][j], novo_tabuleiro[i + di][j + dj] = (
                novo_tabuleiro[i + di][j + dj],
                novo_tabuleiro[i][j],
            )
            filhos.append(
                Estado(
                    novo_tabuleiro,
                    movimento_anterior=(numero_movido, direcao),
                    custo_g=self.custo_g + 1,
                    pai=self,
                )
            )

        return filhos

    def eh_objetivo(self):
        return self.tabuleiro == [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def caminho_desde_inicio(self):
        caminho = []
        estado = self
        while estado.pai is not None:
            caminho.append((estado.movimento_anterior, estado.tabuleiro))
            estado = estado.pai
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
        estado_atual = heapq.heappop(fronteira)

        if estado_atual.eh_objetivo():
            return estado_atual

        tabuleiro_tuple = tuple(tuple(linha) for linha in estado_atual.tabuleiro)
        if tabuleiro_tuple in explorados:
            continue

        explorados.add(tabuleiro_tuple)

        for filho in estado_atual.gerar_filhos():
            filho_tuple = tuple(tuple(linha) for linha in filho.tabuleiro)
            if filho_tuple not in explorados:
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
        if iteracao > 1000:
            break

    return None


# Estado inicial e objetivo

if __name__ == "__main__":
    tabuleiro_inicial = [[1, 8, 2], [0, 4, 3], [7, 6, 5]]
    estado_inicial = Estado(tabuleiro_inicial)
    solucao = a_estrela(estado_inicial)

    if solucao:
        print("\n=== Solução encontrada ===")
        print(f"Custo total: {solucao.custo_g} movimentos")
        print("\nPasso a passo:")
        print("Estado inicial:")
        print("\n".join(" ".join(map(str, linha)) for linha in tabuleiro_inicial))

        for movimento, tabuleiro in solucao.caminho_desde_inicio():
            num, dir = movimento
            print(f"\nMovimento: {num} {dir}")
            print("\n".join(" ".join(map(str, linha)) for linha in tabuleiro))

        print("\nEstado final:")
        print(solucao)
    else:
        print("Não foi encontrada solução.")
