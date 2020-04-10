import math
import pandas as pd
import numpy as np
from logger import CustomLogger
from funcoes import func_logistica, pos_proc
from timer import timerfunc
from plotter import my_plot
from decimal import *

getcontext().prec = 6

np.set_printoptions(suppress=True)
ROUND = 8
MAX_EPOCAS = 10000
PRECISAO = 0.000001
N = 0.01
eqm_anterior = 0.

df = pd.read_csv('data/exe_06_treinamento.csv', sep=',', header=None)
X = df.iloc[0:129, [0, 1, 2, 3]].values.astype(np.longdouble)
X = np.insert(X, 0, -1, axis=1)
Y = df.iloc[0:129, [4, 5, 6]].values

# df = pd.read_csv('data/exe_06_treinamento.csv', sep=',', header=None)
# X = df.iloc[0:17, [0, 1, 2, 3]].values.astype(np.longdouble)
# X = np.insert(X, 0, -1, axis=1)
# Y = df.iloc[0:17, [4, 5, 6]].values

@timerfunc
def treino(X, Y, max_epoca=3000, seed=0):
    np.random.seed(seed)
    w1 = np.random.uniform(0, 1, (15, 5)).astype(np.longdouble)
    w2 = np.random.uniform(0, 1, (3, 15)).astype(np.longdouble)
    w1_inicial = np.copy(w1)
    w2_inicial = np.copy(w2)
    w1_final = None
    w2_final = None
    eqm_hist = []
    eqm_atual = 0.
    eqm_anterior = 0.
    epoca = 0
    while epoca < MAX_EPOCAS:
        eqm_anterior = eqm_atual

        i_1 = np.ones(shape=(1, 15))
        y_1 = np.ones(shape=(1, 15))
        i_2 = np.ones(shape=(1, 3))
        y_2 = np.ones(shape=(1, 3))

        for k, amostras in enumerate(X):
            for j in range(15):
                i_1[0][j] = np.dot(amostras, w1[j])
                y_1[0][j] = func_logistica(i_1[0][j])

            for j in range(3):
                i_2[0][j] = np.dot(y_1, w2[j])
                y_2[0][j] = func_logistica(i_2[0][j])

            eqm_atual += np.sum(0.5 * (Y[k] - y_2) ** 2) / 3.
            delta_2 = np.around(((Y[k] - y_2) * (y_2 * (1 - y_2))), ROUND)

            for j in range(3):
                w2[j] = np.around((w2[j] + N * delta_2[0][j] * y_1), ROUND)

            delta_1 = np.ones(shape=(3, 15))
            for j1 in range(15):
                for j2 in range(3):
                    delta_1[j2] = delta_2[0][j2] * w2[j2] * (y_1 * (1 - y_1))

            delta_1 = (delta_1[0] + delta_1[1] + delta_1[2]) / 3.
            for n in range(15):
                w1[n] = np.around((w1[n] + N * delta_1[n] * amostras), ROUND)

        eqm_atual = eqm_atual / len(X)
        eqm_hist.append(eqm_atual)
        epoca += 1

        if abs(Decimal(round(eqm_atual, 6)) - Decimal(round(eqm_anterior, 6))) <= Decimal(PRECISAO):
            print("Precisão atingida!!!")
            break
    w1_final = w1
    w2_final = w2
    return w1_inicial, w2_inicial, w1_final, w2_final, eqm_hist, epoca

def teste(X, Y, w1, w2):
    i_1 = np.ones(shape=(1, 15))
    y_1 = np.ones(shape=(1, 15))
    i_2 = np.ones(shape=(1, 3))
    y_2 = np.ones(shape=(1, 3))
    y_2_pos = np.ones(shape=(1, 3))
    resultados_pre = []
    resultados_pos = []
    acertos = np.ones(shape=(1, 18))
    for k, amostras in enumerate(X):
        for j in range(15):
            i_1[0][j] = np.dot(amostras, w1[j])
            y_1[0][j] = func_logistica(i_1[0][j])

        for j in range(3):
            i_2[0][j] = np.dot(y_1, w2[j])
            y_2[0][j] = func_logistica(i_2[0][j])
            y_2_pos[0][j] = pos_proc(y_2[0][j])
        
        resultados_pre.append(np.copy(y_2))
        resultados_pos.append(np.copy(y_2_pos))
        r = Y[k] == y_2_pos
        for _ in range(3):
            if r[0][_] == False:
                acertos[0][k] = 0
                break
    return resultados_pos, resultados_pre, acertos

def execute_n_vezes(n, seed=0):
    for i in range(n):
        seed += 1
        print("\n")
        print('---- Experimento ---- {}'.format(i + 1))
        print('Seed: {}'.format(seed))
        w1_inicial, w2_inicial, w1_final, w2_final, eqm_hist, epoca = treino(
            X, Y, MAX_EPOCAS, seed=seed)
        print('Pesos w1 Inicial: {}'.format(w1_inicial.tolist()))
        print('Pesos w2 Inicial: {}'.format(w2_inicial.tolist()))
        print('Pesos w1 final: {}'.format(w1_final.tolist()))
        print('Pesos w2 final: {}'.format(w2_final.tolist()))
        print('Épocas: {}'.format(epoca))
        print('EQM Histórico: {}'.format(eqm_hist))
        my_plot(i, eqm_hist)

if __name__ == "__main__":
    log = CustomLogger()
    execute_n_vezes(5, 74)
#     w1_teste = np.array([[ 1.50537272,  1.19875256,  1.34305235,  1.10501967,  0.97454297],
#  [ 3.84395282,  2.56302574,  3.59411068,  3.01198431,  2.38528752],
#  [ 2.70906912, 0.90601301,  1.09831305,  1.42989091,  0.77783728],
#  [-1.62153746,  0.37947668,  1.15541458,  1.10540204,  1.1012172 ],
#  [ 4.76705425,  1.80722737,  1.74610432,  2.0128042,   1.76472005],
#  [ 1.12523425,  0.18517864,  1.05119248,  0.60936221,  0.56926745],
#  [-0.23106765,  0.5174877,   0.10964228,  0.26124113, -0.24014518],
#  [ 3.49322318,  1.0442477,   1.23007046,  1.51919535,  1.55927098],
#  [-0.89068095, -0.39301458, -0.2993283,  -0.9214896,  -0.29984736],
#  [-0.44723539, -0.80052098, -1.3114839,  -0.72334412, -0.53100797],
#  [ 0.44949445,  0.56369319,  1.23499658,  0.20073089,  0.24749487],
#  [-0.79412088, -0.21214858, -0.97515196, -0.50558668, -0.56715835],
#  [-1.18209697, -0.79995086, -0.46202398, -0.9661237,  -0.88661376],
#  [ 0.14739821,  0.62061288, -0.163845,    0.60260436, -0.10995835],
#  [ 3.51845983,  0.94954086,  1.52999539,  1.26935671,  1.64192381]])
#     w2_teste = np.array([[-2.66824961, -3.94116009, -2.86954281,  1.83224689, -4.20597793, -1.38417487,
#    0.81811638, -3.54454429,  2.37164323,  2.93707599, -0.76338298,  2.79249293,
#    2.95501272,  0.74111274, -3.05898073],
#  [ 0.68038484,  8.67236173, -1.98281093, -0.11614912, -5.0167709,  -0.76929182,
#   -0.8477312,  -3.31580902, -0.63084584, -2.23016003,  0.11229503, -1.44697251,
#   -0.75127125, -0.68434468, -3.13843203],
#  [ 1.42479593,  1.4889081,   2.60458046, -4.14580501,  5.24227081,  0.49471328,
#   -1.29145618,  3.34488415, -3.21934632, -2.40590382, -0.48102742, -2.93051918,
#   -4.09930109, -0.56994566,  3.92179377]])
#     resultados_pos, resultados_pre, acertos = teste(X, Y, w1_teste, w2_teste)
#     print('Resultado pós: {}'.format(resultados_pos))
#     print('Resultado pré: {}'.format(resultados_pre))
#     print('Resultado acertos: {}'.format(acertos))
    log.finish()
