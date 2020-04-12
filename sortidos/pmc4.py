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


ROUND = 8
log = CustomLogger()
np.set_printoptions(suppress=True)
seed = 3
print('Seed {}'.format(seed))
np.random.seed(seed)
EPOCAS = 800
PRECISAO = 0.000001
N = 0.01
df = pd.read_csv('data/exe_05_treinamento.csv', sep=',', header=None)
X = df.iloc[0:199, [0, 1, 2]].values.astype(np.float128)
X = np.insert(X, 0, -1, axis=1)
Y = df.iloc[0:199, [-1]].values
w1 = np.around(np.random.uniform(0, 1, (3, 4)).astype(np.float128), ROUND)
w2 = np.around(np.random.uniform(0, 1, (1, 4)).astype(np.float128), ROUND)
print('Peso w1 inicial: {}'.format(str(w1)))
print('Peso w2 inicial: {}'.format(w2))


def func_logistica(e): return 1. / (1. + np.exp(-e))


def derivada(e): return e * (1.0 - e)


logistica = np.vectorize(func_logistica)
epoca = 0
eqm_hist = []
eqm_atual = 0.
eqm_anterior = 0.
eq_atual = 0.
eq_anterior = 0.

while epoca < EPOCAS:
    eqm_anterior = eqm_atual
    for k, amostras in enumerate(X):
        i_1 = np.around(np.sum(amostras * w1, axis=1), ROUND)
        y_1 = np.around(logistica(i_1), 6)
        y_1 = np.around(np.insert(y_1, 0, -1, axis=0), ROUND)
        i_2 = np.around(np.sum((y_1 * w2), axis=1), ROUND)
        y_2 = np.around(logistica(i_2), ROUND)
        eqm_atual += 0.5 * (Y[k] - y_2) ** 2
        delta_2 = np.around(((Y[k] - y_2) * i_2 * (1 - i_2)), ROUND)
        w2 = np.around((w2 - N * delta_2 * y_1), ROUND)
        delta_1 = - np.sum(delta_2 * w2) * y_1 * (1 - y_1)
        delta_1 = np.around(delta_1, ROUND)
        w1[0] = np.around((w1[0] + N * delta_1 * amostras), ROUND)
        w1[1] = np.around((w1[1] + N * delta_1 * amostras), ROUND)
        w1[2] = np.around((w1[2] + N * delta_1 * amostras), ROUND)

    eqm_atual = np.around((eqm_atual / len(X)), ROUND)
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
