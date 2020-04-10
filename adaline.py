
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


def rede_adaline_treinamento_v1(
        amostras,
        pesos,
        rotulos,
        TAXA_DE_APRENDIZAGEM,
        EPOCAS,
        TAXA_DE_ERRO):
    pesos_iniciais = np.copy(pesos)
    historico_eqm = [0]
    plot = [0]
    melhores_pesos = np.copy(pesos)
    eqm_atual = 0.000000
    for epoca in range(EPOCAS):
        for i, entradas in enumerate(amostras):
            eqm_anterior = eqm_atual
            u = np.around(np.dot(entradas, pesos), 6)
            d = rotulos[i].item()
            eqm_atual = (d - u) ** 2
            historico_eqm.append(eqm_atual)
            for j, peso in enumerate(pesos):
                pesos[j] = peso + (TAXA_DE_APRENDIZAGEM *
                                   (rotulos[i] - u)) * entradas[j]
        eqm_atual = sum(historico_eqm) / len(amostras)
        plot.append(eqm_atual)
        historico_eqm = [0]
        if abs(eqm_atual - eqm_anterior) <= TAXA_DE_ERRO:
            break
        elif eqm_atual > eqm_anterior:
            pesos = melhores_pesos

    return pesos_iniciais, pesos, epoca, plot


def rede_adaline_classificacao_v0(amostras, pesos, func_de_ativacao):
    classificacao_A = np.zeros(shape=(1, amostras.shape[1]))
    classificacao_B = np.zeros(shape=(1, amostras.shape[1]))
    for i, entradas in enumerate(amostras):
        produto = np.dot(entradas, pesos)
        y_saida = func_de_ativacao(produto)
        entradas_aux = np.copy(entradas)
        entradas_aux[0] = i + 1
        if y_saida == 1:
            classificacao_B = np.vstack((classificacao_B, entradas_aux))
        else:
            classificacao_A = np.vstack((classificacao_A, entradas_aux))

    return classificacao_A, classificacao_B
