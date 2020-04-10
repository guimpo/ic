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
PRECISAO = 0.5 * 0.000001
N = 0.01
ALFA = 0.8


def momentum(w_atual, w_anterior, a=ALFA):
    return a * np.subtract(w_atual,  w_anterior)


@timerfunc
def treino(X, Y, max_epoca, x, neurons_1, neurons_2, seed):
    np.random.seed(seed)
    w1 = np.random.uniform(0, 1, (neurons_1, x + 1)).astype(np.longdouble)
    w2 = np.random.uniform(
        0, 1, (neurons_2, neurons_1 + 1)).astype(np.longdouble)
    w1_inicial = np.copy(w1)
    w2_inicial = np.copy(w2)
    w1_final = None
    w2_final = None
    eqm_hist = []
    eqm_atual = 0.
    eqm_anterior = 0.
    epoca = 0
    w1_anterior = np.copy(w1_inicial)
    w1_atual = np.copy(w1_anterior)
    w2_atual = np.copy(w2_inicial)
    w2_anterior = np.copy(w2_inicial)

    while epoca < max_epoca:
        eqm_anterior = eqm_atual

        for k, amostras in enumerate(X):
            i_1 = np.empty(shape=(1, neurons_1 + 1))
            y_1 = np.empty(shape=(1, neurons_1))
            i_2 = np.empty(shape=(1, neurons_2))
            y_2 = np.empty(shape=(1, neurons_2))

            for j in range(neurons_1):
                i_1[0][j] = np.dot(amostras, w1[j])
                y_1[0][j] = func_logistica(i_1[0][j])
            y_1 = np.insert(y_1, 0, -1, axis=1)

            for j in range(neurons_2):
                i_2[0][j] = np.dot(y_1, w2[j])
                y_2[0][j] = func_logistica(i_2[0][j])

            eqm_atual += (0.5 * (Y[k] - y_2) ** 2).item()
            delta_2 = np.around(((Y[k] - y_2) * (y_2 * (1 - y_2))), ROUND)

            momentum_2 = momentum(w2_atual, w2_anterior, ALFA)
            w2_anterior = np.copy(w2)
            for j in range(neurons_2):
                w2[j] = np.around(
                    (w2[j] + momentum_2[j] + N * delta_2[0][j] * y_1), ROUND)
            w2_atual = np.copy(w2)

            delta_1 = np.empty(shape=(neurons_2, neurons_1 + 1))
            for j1 in range(neurons_1):
                delta_1 = delta_2 * w2 * (y_1 * (1 - y_1))

            momentum_1 = momentum(w1_atual, w1_anterior, ALFA)
            w1_anterior = np.copy(w1)
            for j in range(neurons_1):
                w1[j] = np.around(
                    (w1[j] + momentum_1[j] + N * delta_1[0][j] * amostras), ROUND)
            w1_atual = np.copy(w1)

        eqm_atual = eqm_atual / len(X)
        eqm_atual = eqm_atual.astype(float)
        eqm_hist.append(eqm_atual)
        epoca += 1

        if abs(Decimal(round(eqm_atual, 8)) - Decimal(round(eqm_anterior, 8))) <= Decimal(PRECISAO):
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


def execute_n_vezes(repetir, x, y, max_epoca, input, neurons_1, neurons_2, seed):
    for i in range(repetir):
        seed += 1
        print("\n")
        print('---- Experimento ---- {}'.format(i + 1))
        print('Seed: {}'.format(seed))
        w1_inicial, w2_inicial, w1_final, w2_final, eqm_hist, epoca = treino(
            X=x,
            Y=y,
            max_epoca=max_epoca,
            x=input,
            neurons_1=neurons_1,
            neurons_2=neurons_2,
            seed=seed)
        print('Pesos w1 Inicial: {}'.format(w1_inicial.tolist()))
        print('Pesos w2 Inicial: {}'.format(w2_inicial.tolist()))
        print('Pesos w1 final: {}'.format(w1_final.tolist()))
        print('Pesos w2 final: {}'.format(w2_final.tolist()))
        print('Épocas: {}'.format(epoca))
        print('EQM Histórico: {}'.format(eqm_hist))
        my_plot(i, eqm_hist)


if __name__ == "__main__":
    # df = pd.read_csv('data/exe_07_treinamento_05.csv', sep=',', header=None)
    # X = df.iloc[0:95, [0, 1, 2, 3, 4]].values.astype(np.longdouble)
    # X = np.insert(X, 0, -1, axis=1)
    # Y = df.iloc[0:95, [5]].values.astype(np.longdouble)
    # log = CustomLogger()
    # execute_n_vezes(
    #     repetir=3,
    #     x=X,
    #     y=Y,
    #     max_epoca=MAX_EPOCAS,
    #     input=5,
    #     neurons_1=10,
    #     neurons_2=1,
    #     seed=0)
    # log.finish()

    # df = pd.read_csv('data/exe_07_treinamento_10.csv', sep=',', header=None)
    # X = df.iloc[0:90, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].values.astype(np.longdouble)
    # X = np.insert(X, 0, -1, axis=1)
    # Y = df.iloc[0:90, [10]].values.astype(np.longdouble)
    # log = CustomLogger()
    # execute_n_vezes(
    #     repetir=3,
    #     x=X,
    #     y=Y,
    #     max_epoca=MAX_EPOCAS,
    #     input=10,
    #     neurons_1=15,
    #     neurons_2=1,
    #     seed=0)
    # log.finish()

    df = pd.read_csv('data/exe_07_treinamento_15.csv', sep=',', header=None)
    X = df.iloc[0:85, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].values.astype(np.longdouble)
    X = np.insert(X, 0, -1, axis=1)
    Y = df.iloc[0:85, [15]].values.astype(np.longdouble)
    log = CustomLogger()
    execute_n_vezes(
        repetir=1,
        x=X,
        y=Y,
        max_epoca=MAX_EPOCAS,
        input=15,
        neurons_1=25,
        neurons_2=1,
        seed=14)
    log.finish()
