import pandas as pd
import numpy as np
from logger import CustomLogger
from perceptron import rede_perceptron_treinamento_v0, rede_perceptron_classificacao_v0, funcao_sinal

log = CustomLogger()
df = pd.read_csv('data/exe_03_data.csv', sep=',', header=None)
entradas = df.iloc[0:29, [0, 1, 2]].values
entradas = np.insert(entradas, 0, -1, axis=1)
rotulos = df.iloc[0:29, [-1]].values
entradas_classificao = pd.read_csv(
    'data/exe_03_classificacao.csv',
    sep=',',
    header=None).values
entradas_classificao = np.insert(entradas_classificao, 0, -1, axis=1)
EPOCAS = 3000
TAXA_DE_APRENDIZAGEM = 0.01


print("\n")
print('---- Experimento ---- 01')
np.random.seed(16)
pesos_1 = np.random.uniform(0, 1, 4)
pesos_iniciais_1, pesos_ajustados_1, epocas_1 = rede_perceptron_treinamento_v0(
    entradas, pesos_1, rotulos, TAXA_DE_APRENDIZAGEM, EPOCAS)
print(pesos_iniciais_1, pesos_ajustados_1, epocas_1)

classificacao_a, classificacao_b = rede_perceptron_classificacao_v0(
    entradas_classificao, pesos_ajustados_1, funcao_sinal)
print("C2 ", classificacao_a)
print("C1 ", classificacao_b)


print("\n")
print('---- Experimento ---- 02')
np.random.seed(17)
pesos_2 = np.random.uniform(0, 1, 4)

pesos_iniciais_2, pesos_ajustados_2, epocas_2 = rede_perceptron_treinamento_v0(
    entradas, pesos_2, rotulos, TAXA_DE_APRENDIZAGEM, EPOCAS)
print(pesos_iniciais_2, pesos_ajustados_2, epocas_2)

classificacao_a, classificacao_b = rede_perceptron_classificacao_v0(
    entradas_classificao, pesos_ajustados_2, funcao_sinal)
print("C2 ", classificacao_a)
print("C1 ", classificacao_b)


print("\n")
print('---- Experimento ---- 03')
np.random.seed(18)
pesos_3 = np.random.uniform(0, 1, 4)

pesos_iniciais_3, pesos_ajustados_3, epocas_3 = rede_perceptron_treinamento_v0(
    entradas, pesos_3, rotulos, TAXA_DE_APRENDIZAGEM, EPOCAS)
print(pesos_iniciais_3, pesos_ajustados_3, epocas_3)

classificacao_a, classificacao_b = rede_perceptron_classificacao_v0(
    entradas_classificao, pesos_ajustados_3, funcao_sinal)
print("C2 ", classificacao_a)
print("C1 ", classificacao_b)


print("\n")
print('---- Experimento ---- 04')
np.random.seed(19)
pesos_4 = np.random.uniform(0, 1, entradas.shape[1])

pesos_iniciais_4, pesos_ajustados_4, epocas_4 = rede_perceptron_treinamento_v0(
    entradas, pesos_4, rotulos, TAXA_DE_APRENDIZAGEM, EPOCAS)
print(pesos_iniciais_4, pesos_ajustados_4, epocas_4)

classificacao_a, classificacao_b = rede_perceptron_classificacao_v0(
    entradas_classificao, pesos_ajustados_4, funcao_sinal)
print("C2 ", classificacao_a)
print("C1 ", classificacao_b)


print("\n")
print('---- Experimento ---- 05')
np.random.seed(20)
pesos_5 = np.random.uniform(0, 1, entradas.shape[1])


pesos_iniciais_5, pesos_ajustados_5, epocas_5 = rede_perceptron_treinamento_v0(
    entradas, pesos_5, rotulos, TAXA_DE_APRENDIZAGEM, EPOCAS)
print(pesos_iniciais_5, pesos_ajustados_5, epocas_5)

classificacao_a, classificacao_b = rede_perceptron_classificacao_v0(
    entradas_classificao, pesos_ajustados_5, funcao_sinal)
print("C2 ", classificacao_a)
print("C1 ", classificacao_b)

print("\n")
print('---- Professor ----')
pesos_prof = np.array([-17.6490, 8.6697, 14.0299, -4.0960])
classificacao_a, classificacao_b = rede_perceptron_classificacao_v0(
    entradas_classificao, pesos_prof, funcao_sinal)
print("C2 ", classificacao_a)
print("C1 ", classificacao_b)

log.finish()
