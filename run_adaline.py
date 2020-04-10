import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from logger import CustomLogger
from adaline import rede_adaline_treinamento_v1, rede_adaline_classificacao_v0, funcao_sinal

now = datetime.datetime.now()
date = datetime.datetime.now()
log = CustomLogger()
df = pd.read_csv('data/exe_04_data.csv', sep=',', header=None)
entradas = df.iloc[0:34, [0, 1, 2, 3]].values
entradas = np.insert(entradas, 0, -1, axis=1)
rotulos = df.iloc[0:34, [-1]].values
entradas_classificao = pd.read_csv(
    'data/exe_04_classificacao.csv',
    sep=',',
    header=None).values
entradas_classificao = np.insert(entradas_classificao, 0, -1, axis=1)

EPOCAS = 5000
TAXA_DE_APRENDIZAGEM = 0.00025
TAXA_DE_ERRO = 0.000001


def execute_n_vezes(n, seed=15):
    for i in range(n):
        seed += 1
        print("\n")
        print('---- Experimento ---- {}'.format(i + 1))
        print('Seed: {}'.format(seed))
        np.random.seed(seed)
        pesos_1 = [-1.85574372, 1.31822575,
                   1.68291399, -0.44537004, -1.21001573]
        #[-1.85582058,  1.31809741,  1.68239973, -0.44647037, -1.20974859]
        #[-1.86084734,  1.32078797,  1.67998505, -0.44558597, -1.20174501]
        #np.random.uniform(0, 1, 5)
        print('Pesos Iniciais: {}'.format(pesos_1))
        pesos_iniciais_1, pesos_ajustados_1, epocas_1, historico_eqm_1 = rede_adaline_treinamento_v1(
            entradas, pesos_1, rotulos, TAXA_DE_APRENDIZAGEM, EPOCAS, TAXA_DE_ERRO)
        print('Pesos Finais  : {}'.format(pesos_ajustados_1))
        print('Épocas: {}'.format(epocas_1))
        plt.autoscale(tight=True)
        plt.ticklabel_format(useOffset=False)
        plt.plot(historico_eqm_1[1:])
        plt.ylabel('EQM')

        plt.xlabel('Épocas')
        plt.savefig('data/exp-{}-{}-{}-{}-{}-{}-{}-eqm-plot.png'.format(i +
                                                                        1, now.year, now.month, now.day, now.hour, now.minute, now.second))
        plt.clf()
        classificacao_a, classificacao_b = rede_adaline_classificacao_v0(
            entradas_classificao, pesos_ajustados_1, funcao_sinal)
        print('Min: {}'.format(min(historico_eqm_1[1:])))
        print("A ", classificacao_a)
        print("B ", classificacao_b)


execute_n_vezes(5)


# -1.8075,1.3103,1.6436,-0.4279,-1.1857

log.finish()
