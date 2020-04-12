import math
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


def funcao_logistica(u_potencial_de_ativacao):
    return 1 / (1 + math.exp(-u_potencial_de_ativacao))


EPOCAS = 61
PRECISAO = 0.000001
N = 0.1

df = pd.read_csv('data/exe_05_treinamento.csv', sep=',', header=None)
X = df.iloc[0:199, [0, 1, 2]].values.astype(np.float128)
X = np.insert(X, 0, -1, axis=1)
Y = df.iloc[0:199, [-1]].values
w1 = np.around(np.random.uniform(0, 1, (3, 4)), 4).astype(np.float128)
w2 = np.around(np.random.uniform(0, 1, (1, 4)), 4).astype(np.float128)


def func_logistica(e): return 1. / (1. + np.exp(-e))


def derivada(e): return e * (1.0 - e)


eqm_atual = 0
epoca = 0
eqm_hist = []

while epoca < EPOCAS:
    eqm_anterior = eqm_atual
    eqm_atual = 0
    for k, amostras in enumerate(X):
        i_1 = np.sum(amostras * w1, axis=1)
        func1 = np.vectorize(func_logistica)
        y_1 = func1(i_1)
        y_1 = np.insert(y_1, 0, -1, axis=0)
        i_2 = np.sum((y_1 * w2), axis=1)
        y_2 = np.around(i_2, 6)
        erro = 0.5 * (Y[k] - y_2) ** 2
        eqm_atual += erro
        g_2 = np.vectorize(derivada)(y_2)
        delta_2 = np.around(erro * g_2, 6)
        w2 = w2 + N * delta_2 * y_2
        g_1 = np.around(np.vectorize(derivada)(i_1), 6)
        g_1 = np.insert(g_1, 0, -1, axis=0)
        delta_1 = - np.sum(delta_2 * w2) * g_1
        w1[0] = w1[0] + N * delta_1 * amostras
        w1[1] = w1[1] + N * delta_1 * amostras
        w1[2] = w1[2] + N * delta_1 * amostras
    eqm_atual = eqm_atual / len(X)
    eqm_hist.append(eqm_atual.item())
    epoca += 1

    # print(eqm_atual - eqm_anterior)
    if abs(eqm_atual - eqm_anterior) <= PRECISAO:
        break
print(w1)
print(w2)
print(epoca)
print(eqm_hist)
