
import pandas as pd
import numpy as np


def funcao_degrau(u_potencial_de_ativacao):
    if(u_potencial_de_ativacao >= 0):
        return 1
    return 0


def funcao_sinal(u_potencial_de_ativacao):
    '''
        retorna -1 ou 1
    '''
    if(u_potencial_de_ativacao >= 0):
        return 1.
    return -1.


def rede_perceptron_treinamento_v0(
        amostras,
        pesos,
        rotulos,
        TAXA_DE_APRENDIZAGEM,
        EPOCAS):
    pesos_iniciais = np.copy(pesos)
    for epoca in range(EPOCAS):
        existe_erro = False
        for i, entradas in enumerate(amostras):
            soma_ponderada = np.dot(entradas, pesos)
            y_saida = funcao_sinal(soma_ponderada)
            if rotulos[i] != y_saida:
                existe_erro = True
                for j, peso in enumerate(pesos):
                    pesos[j] = peso + (TAXA_DE_APRENDIZAGEM *
                                       (rotulos[i] - y_saida)) * entradas[j]
        if not existe_erro:
            break
    print("pesos iniciais", pesos_iniciais)
    print("pesos finais", pesos)
    return pesos_iniciais, pesos, epoca


def rede_perceptron_classificacao_v0(amostras, pesos, func_de_ativacao):
    classificacao_C2 = np.array(['Idx', 'limiar', 'x1', 'x2', 'x3'])
    classificacao_C1 = np.array(['Idx', 'limiar', 'x1', 'x2', 'x3'])

    for i, entradas in enumerate(amostras):
        produto = np.dot(entradas, pesos)
        y_saida = func_de_ativacao(produto)

        if y_saida == 1.:
            classificacao_C2 = np.vstack(
                (classificacao_C2, np.append(int(i + 1), entradas)))
        else:
            classificacao_C1 = np.vstack(
                (classificacao_C1, np.append(int(i + 1), entradas)))
    return classificacao_C2, classificacao_C1
