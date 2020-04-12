import math
import pandas as pd
import numpy as np
from logger import CustomLogger


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


log = CustomLogger()
np.set_printoptions(suppress=True)
seed = 9
print('Seed {}'.format(seed))
np.random.seed(seed)
EPOCAS = 200
PRECISAO = 0.000001
N = 0.1

df = pd.read_csv('data/exe_05_treinamento.csv', sep=',', header=None)
X = df.iloc[0:199, [0, 1, 2]].values.astype(np.float128)
X = np.insert(X, 0, -1, axis=1)
Y = df.iloc[0:199, [-1]].values
w1 = np.random.uniform(0, 1, (3, 4)).astype(np.float128)
w2 = np.random.uniform(0, 1, (1, 4)).astype(np.float128)
print('Peso w1 inicial: {}'.format(str(w1)))
print('Peso w2 inicial: {}'.format(w2))


def func_logistica(e): return 1. / (1. + np.exp(-e))


def derivada(e): return e * (1.0 - e)


logistica = np.vectorize(func_logistica)
epoca = 0
eqm_hist = []
eqm_atual = 0
eqm_anterior = 0
eq_atual = 0
eq_anterior = 0

while epoca < EPOCAS:
    eqm_anterior = eqm_atual
    for k, amostras in enumerate(X):
        eq_anterior = eq_atual
        i_1 = np.sum(amostras * w1, axis=1)
        y_1 = logistica(i_1)
        y_1 = np.insert(y_1, 0, -1, axis=0)
        i_2 = np.sum((y_1 * w2), axis=1)
        y_2 = logistica(i_2)
        eq_atual = 0.5 * (Y[k] - y_2) ** 2
        eqm_atual += eq_atual
        if eq_atual > eq_anterior:
            delta_2 = (Y[k] - y_2) * i_2 * (1 - i_2)
            w2 = w2 - N * delta_2 * y_1
            delta_1 = - np.sum(delta_2 * w2) * y_1 * (1 - y_1)
            w1[0] = w1[0] + N * delta_1 * amostras
            w1[1] = w1[1] + N * delta_1 * amostras
            w1[2] = w1[2] + N * delta_1 * amostras
        eq_anterior = eq_atual
    eqm_atual = eqm_atual / len(X)
    eqm_hist.append(eqm_atual.item())
    epoca += 1

    if abs(eqm_atual - eqm_anterior) <= PRECISAO:
        break
print('Peso w1 final: {}'.format(w1))
print('Peso w2 final: {}'.format(w2))
print('Época: {}'.format(epoca))
print('EQM : {}'.format(eqm_hist[-1]))
print('EQM Histórico: {}'.format(eqm_hist))
log.finish()
