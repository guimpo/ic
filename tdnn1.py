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
    results = []
    eq_hist = np.empty(shape=(1, 20))
    eq = 0

    for k, amostras in enumerate(X):
        i_1 = np.empty(shape=w1.shape[0])
        y_1 = np.empty(shape=w1.shape[0])
        i_2 = np.empty(shape=w2.shape[0])
        y_2 = np.empty(shape=w2.shape[0])

        for j in range(w1.shape[0]):
            i_1[j] = np.dot(amostras, w1[j])
            y_1[j] = func_logistica(i_1[j])
        y_1 = np.insert(y_1, 0, -1, axis=0)

        for j in range(w2.shape[0]):
            i_2[j] = np.dot(y_1, w2[j])
            y_2[j] = func_logistica(i_2[j])

        results.append(y_2.item())
        eq =  ((Y[k] - y_2) ** 2).item()
        eq_hist[0][k] = eq
    print('Erro relativo médio: {}'.format(eq_hist.mean()))
    print('Variância: {}'.format(eq_hist.var()))
    return eq_hist, results

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
    # Rede 01
    # df = pd.read_csv('data/exe_07_treinamento_rede_01.csv', sep=',', header=None)
    # X = df.iloc[0:95, [0, 1, 2, 3, 4]].values.astype(np.longdouble)
    # X = np.insert(X, 0, -1, axis=1)
    # Y = df.iloc[0:95, [5]].values.astype(np.longdouble)
    # log = CustomLogger()
    # execute_n_vezes(
    #      repetir=3,
    #      x=X,
    #      y=Y,
    #      max_epoca=MAX_EPOCAS,
    #      input=5,
    #      neurons_1=10,
    #      neurons_2=1,
    #      seed=0)
    # log.finish()

    # df = pd.read_csv('data/exe_07_teste_rede_01.csv', sep=',', header=None)
    # X = df.iloc[0:20, [0, 1, 2, 3, 4]].values.astype(np.longdouble)
    # X = np.insert(X, 0, -1, axis=1)
    # Y = df.iloc[0:20, [5]].values.astype(np.longdouble)
    # log = CustomLogger()   
    
    # w1 = np.array([[0.35642003, 2.10630885, -1.32572208, 3.47431068, -0.36078935, 1.1702409], [1.50269981, -0.03362593, -0.04421398, 0.69702599, 0.2443993, 0.27067782], [0.24785066, 0.82025556, 0.0547564, 0.52645083, 0.40584414, 0.51573995], [0.31778098, 0.21671064, 0.66954762, 1.21917613, 0.2859228, 0.68047772], [1.2254051, 0.9560389, -0.19206152, 0.54706449, 0.10146549, 0.87138058], [0.17536486, 0.38713226, 0.93929352, 0.52367262, 0.6830494, 0.28029089], [0.59884671, 0.77176387, 0.13479795, 0.42798662, 0.98991654, 0.72554707], [0.43917579, 0.79242846, -0.00361556, 0.6540058, 0.89132357, 0.27036544], [0.24728324, 0.0186316, 0.15293888, 0.27212974, 0.21211021, 0.19833983], [0.52754839, 0.00045714, 0.60215546, -0.02608344, 0.56456183, 0.66454965]])
    # w2 = np.array([[-0.31014629, -1.7258565, 0.01883966, -0.3300445, -0.76439505, -0.12961403, 0.52061505, -0.37447987, 0.2942029, 0.21614017, 1.05772475]])
    
    # w1 = np.array([[-0.77526472, 0.84349955, 0.54601133, 1.75210167, 0.62654937, 1.0572173], [0.2475269, 0.59969892, 0.29013069, 0.24276002, 0.60944606, 0.51108546], [0.20893134, 0.47295586, 0.17503258, 0.72614751, 0.83684703, 0.45742639], [0.84812869, 0.08687383, 0.4972931, 0.08130951, 0.42416436, 0.10215798], [0.15198344, 0.58491613, 0.22119044, 0.09055572, 0.21364121, 0.3388964], [0.51636647, 0.17775587, 0.63194724, 0.44947125, 0.49278952, 0.36468242], [0.8216039, 0.5683386, 0.15530111, 0.68657527, 0.95650072, 0.48901844], [0.87536278, 0.35667702, 0.56169973, 0.45491511, 0.43654101, 0.7894124], [0.65588093, 0.88601888, 0.53151551, -0.01875311, 0.33972789, 0.78951973], [0.44424334, 0.00976164, 0.23927674, 0.04371961, 0.98352144, 0.95432284]])
    # w2 = np.array([[1.36516275, 0.15399514, 0.37386503, -0.28074577, 0.08787926, 0.16434939, 0.01815122, -0.32983319, 0.66010551, 0.08251779, 0.17047585]])
    
    # w1 = np.array([[0.4970291, 2.63171514, -1.60050383, 4.37729034, -0.13562838, 2.51509029], [1.72746434, -0.27119076, -0.45221602, 0.44600372, -0.22994778, -0.06262431], [0.70397783, 0.30746392, 0.61118604, 0.71915538, -0.00107262, 0.5701376], [0.28585045, 0.34100372, 0.33904446, 0.50611849, 0.44798735, 0.10370223], [0.61434614, 0.80585403, 0.23559199, 0.34086076, 0.35623941, 0.94603788], [1.03383494, 0.66512834, 0.87203979, 0.8671276, 0.35857025, 0.08072706], [0.77890889, 0.70495138, 0.13213428, 0.8602729, 0.37614556, 0.52879947], [0.35748727, 0.33445679, 0.35792327, 1.10510195, 0.5682001, 0.29078191], [0.14361081, 0.30453193, 0.43550633, -0.28931808, 0.5950099, 0.0018409], [0.3708456, 0.31442796, 0.82359875, 0.34890076, 0.66958953, 0.70815047]])
    # w2 = np.array([[-0.47192926, -1.72500556, -0.15688101, 0.02977074, -0.08422905, 0.00575331, -0.7100568, -0.36824115, 0.14715703, 0.81846335, 0.29123008]])


    # erro_hist, predicoes = teste(X, Y, w1, w2)
    # print('Histórico erro: {}'.format(erro_hist))
    # print('Predições: {}'.format(predicoes))
    # log.finish()
    

    # Rede 02
    # df = pd.read_csv('data/exe_07_treinamento_rede_02.csv', sep=',', header=None)
    # X = df.iloc[0:90, [x for x in range(10)]].values.astype(np.longdouble)
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

    # df = pd.read_csv('data/exe_07_teste_rede_02.csv', sep=',', header=None)
    # X = df.iloc[0:20, [x for x in range(10)]].values.astype(np.longdouble)
    # X = np.insert(X, 0, -1, axis=1)
    # Y = df.iloc[0:20, [10]].values.astype(np.longdouble)
    # log = CustomLogger()   
    
    # w1 = np.array([[0.42436982, 1.79893146, -1.02736653, 2.88501306, -0.14152394, 0.71125466, -2.16417157, 0.24364116, 0.06279219, 2.06491895, 1.67298502], [2.38279219, -0.37101168, 0.28244134, -0.01258054, 0.59686765, -0.41845108, -0.6151377, -1.0882823, -0.11174312, 0.34177682, 1.50468145], [0.2674065, 0.6814134, 0.9152776, 0.88740562, 0.13128533, 0.01814956, 0.20648572, 0.85524955, 0.14634122, 0.40467686, 1.01881702], [0.62571127, 0.67661423, 0.27709055, 0.7032375, 0.81925366, -0.03127597, 0.67847901, 0.92434198, 0.74164193, 0.2638969, 0.82992207], [0.06088424, 0.42586789, 0.9557965, 0.20413766, 0.29472376, 0.14920113, 0.12170003, 0.73308545, 0.20929448, 0.23240313, 0.40764589], [0.07780413, 0.58709026, 0.11992827, 0.62454361, 0.67865822, 0.11719041, 0.36213047, 0.69907608, 0.38852196, 0.07422263, 0.53039932], [0.51279432, 0.43782071, 1.11015039, 0.28316117, 0.93634641, 0.1822758, 0.50190871, 0.96927361, 0.41554274, 0.04140689, 0.67743664], [0.3160817, 0.69462699, 0.79656232, 0.72936983, 0.63604307, 0.76752346, 0.49360692, 0.33253671, 0.87927269, 0.36190556, 0.83259196], [0.7860612, 0.62022865, 0.04221999, 1.0059639, 0.4129812, 0.53760963, 0.28128693, 0.16710985, 0.8730465, 0.58271214, 0.0434185], [0.75049199, 0.30738563, 0.45748547, 0.95572865, 0.35591763, 0.82860486, 0.48414585, -0.11586663, 0.93464774, 0.68006598, 1.11919385], [0.19128471, 0.09666644, 0.95548168, 0.63090368, 0.07982929, 0.71664757, 0.81054263, 0.89608018, 0.73255933, 0.07278427, 0.01197988], [0.11628334, -0.08553743, 0.29547821, 0.54535362, 0.46389154, 0.56649176, 1.05295721, 0.23019423, 0.17256644, 0.44823113, 0.61134534], [0.63514825, 0.04092731, 0.73103867, 0.32291314, 0.78006632, 0.37284444, 0.74015699, 0.69849302, 0.53399351, 0.17475557, 0.11259101], [0.08101652, 0.03611247, 0.14730911, 0.11940396, 0.67301479, 0.64371292, 0.10419926, 0.20778817, 0.89169537, 0.56267965, 0.00262055], [0.31320802, 0.75160153, 0.14808113, 0.63921322, 0.95401585, 0.82926253, 0.15203298, 0.45000984, 0.60643363, 0.84742866, 0.20092057]])
    # w2 = np.array([[-0.10076472, -2.8684264, -0.47835176, -0.2230979, 0.15721229, 0.13790894, 0.54784776, 0.35959326, -0.2290599, -0.36337768, -0.25424896, 1.0363065, -0.11123565, 0.66057528, -0.08153971, 0.04420211]])
    
    # w1 = np.array([[0.77043198, 5.37414399, -7.3483442, 13.25757938, -1.35566324, 0.30223374, -9.57347287, -1.70066098, 3.31112088, 5.71141632, 15.58135226], [1.18099613, -0.62997903, 0.6320393, -0.19223803, 1.04108315, 0.14374569, 0.26639616, -0.02435621, 0.37974939, 0.29696821, 0.49436193], [0.34644258, 0.01325498, 0.4061427, 0.3617992, 0.30597313, 0.2591349, 0.44894478, 0.43664377, 0.25143833, 0.19025876, -0.0900104], [0.62114104, 0.62469318, -0.04961067, 1.11363394, 0.45450067, 0.00133662, 0.42655749, 0.8606392, 0.75100778, 0.93890783, 1.22611826], [0.87248198, 0.30803045, 0.29727613, 0.63294493, 0.40239993, 0.7531288, 0.43413097, 0.02768404, 0.45344906, 0.72756888, 0.70094662], [0.20141135, 0.18826685, -0.17434724, 1.04050539, 0.9088117, 0.68263758, 0.44868191, 0.69778393, 0.26516954, 0.30229701, 0.96556615], [0.41593671, -0.05549002, 0.96910493, 0.36026177, 0.50056485, 0.27786481, 0.24244928, 0.40562373, 0.84342245, 0.75642488, 0.44618506], [0.08621489, 0.89923911, 0.08733504, 0.77725325, -0.05430163, 0.65650343, 0.21088216, 0.45855817, 0.44962821, 0.40583603, 0.79450277], [1.68507168, -0.01795498, -0.03364936, 0.05612249, 0.47300045, 0.10864692, -0.32183652, 0.09748071, 0.84750089, 0.50692835, 0.91296993], [0.7682388, 0.76599554, 0.06957653, 0.84494212, 0.72777314, 0.79700003, 0.50091094, 0.41009354, 1.03893321, 0.69147246, 0.72787737], [0.42858586, 0.81153845, 0.31321759, 0.44467434, 0.55023836, 0.6096146, 0.61482634, 0.11024167, 0.78850721, 0.39299713, 0.9577489], [0.51010331, 0.15862426, 0.40057973, -0.14312417, 0.35580801, 0.11834902, 0.88041853, 0.48405164, 0.14391873, 0.06590874, 0.01244124], [0.22917597, -0.47383482, 1.81119262, -1.19157174, 1.13149773, 1.22117097, 1.22952073, 0.53180787, -0.2213252, -0.44663149, -2.11861167], [1.02649193, 0.12421531, 1.87825923, -0.4595771, 0.04206392, -0.27788, 1.32437152, -0.55182165, -0.40048354, -0.56902951, -2.07761711], [-0.7200658, 0.26573638, 0.66094152, 0.12440601, 0.95899635, 0.61762987, 1.67554622, 0.74812461, 0.54519526, 0.14979378, -0.27142226]])
    # w2 = np.array([[0.79339073, -0.1673239, 0.44131292, -0.69281552, -1.02421348, -0.70684088, -0.51220236, -0.27742061, -1.2586648, -0.96418349, -0.85740306, -0.17862522, 1.81696201, 3.59138773, 1.6337684, 1.03619764]])

    # w1 = np.array([[-0.54532695, 2.32778162, -0.46064504, 3.33646876, 0.65208683, 2.09173984, -1.75686292, 0.86437291, -0.09061041, 2.3324649, 1.14611226], [0.4538387, 0.64077929, 0.28533097, 0.65949003, 0.59377379, 0.01592452, 0.58112505, 0.25671667, 0.42248258, 0.27142944, 0.68814823], [0.44696018, 0.18722584, 0.51409659, 0.83961612, 0.2885361, 0.24126931, 0.32852878, 0.94177462, 0.96012585, 0.70888262, 0.91915923], [0.87047337, 0.38976782, 0.0678406, 0.68256787, 0.54009232, 0.36767906, 0.18002128, 0.40411382, 0.45222938, 0.28477181, 0.29530034], [0.52983346, 0.79034876, 0.60416758, 0.1689717, 0.27619986, 0.39873299, 0.26561596, 0.16631922, 0.51059227, 0.00669518, 0.43222226], [0.42865079, 0.63997016, 0.76620869, 0.57812544, 0.68988431, 0.31945996, 0.73067785, 0.30705178, 0.5716338, 0.24638298, 0.39610699], [0.07299681, 0.24926638, 0.95909785, 0.24625055, 0.68245611, 0.65324955, 0.70212609, 0.47376572, 0.58904962, 0.07533001, 0.07473333], [0.19861842, 0.15982474, 0.09339486, 0.14432813, 0.54881614, 0.19286474, 0.93883639, 0.68397787, 0.53836311, 0.71672351, 0.268034], [0.94941119, 0.82870135, 0.72226955, 0.46621202, 0.83637232, 0.73443622, 0.65895665, 0.90568477, 0.62821873, 0.35435457, 0.54360561], [0.22409932, 0.17615488, 0.72349722, 0.7614089, 0.96661872, 0.83590212, 0.5481968, 0.07884052, 0.48395911, 0.90948162, 0.7744764], [0.50283038, 0.47660312, 0.18950802, 0.21668274, 0.05480802, 0.90360123, 0.59007772, 0.1460283, 0.39741065, 0.07298594, 0.21553187], [0.80600865, 0.57680384, 0.49789681, 0.15059098, 0.65105808, 0.89842677, 0.50173768, 0.03355839, 0.05421552, -0.00490023, 0.9054492], [0.57897477, 0.8092962, 0.17091205, 0.21869115, 0.68746316, 0.87312753, 0.35500798, 0.04756861, 0.85829795, 0.25906599, 0.72138971], [0.19832047, 0.65252415, 0.85344311, 0.65835982, 0.36037012, 0.69410207, 0.17861607, 0.29278944, 0.42017458, 0.82721416, 0.12615724], [0.91114959, 0.81025246, 0.94387083, 0.39823041, 0.18549901, 0.24658044, 0.77723224, 0.96105496, 0.67928262, 0.4974818, 0.29064915]])
    # w2 = np.array([[1.34142017, 0.25538873, -0.36586272, -0.32321272, 0.3487857, 0.44277081, -0.11121147, -0.03776431, 0.08456941, 0.08469629, -0.4243842, 0.48856345, 0.13322233, 0.29169896, -0.3482366, 0.59148081]])
    
    # erro_hist, predicoes = teste(X, Y, w1, w2)
    # print('Histórico erro: {}'.format(erro_hist))
    # print('Predições: {}'.format(predicoes))
    # log.finish()


    # Rede 03
    # df = pd.read_csv('data/exe_07_treinamento_rede_03.csv', sep=',', header=None)
    # X = df.iloc[0:85, [x for x in range(15)]].values.astype(np.longdouble)
    # X = np.insert(X, 0, -1, axis=1)
    # Y = df.iloc[0:85, [5]].values.astype(np.longdouble)
    # log = CustomLogger()
    # execute_n_vezes(
    #      repetir=1,
    #      x=X,
    #      y=Y,
    #      max_epoca=MAX_EPOCAS,
    #      input=15,
    #      neurons_1=25,
    #      neurons_2=1,
    #      seed=0)
    # execute_n_vezes(
    #      repetir=1,
    #      x=X,
    #      y=Y,
    #      max_epoca=MAX_EPOCAS,
    #      input=15,
    #      neurons_1=25,
    #      neurons_2=1,
    #      seed=14)
    # execute_n_vezes(
    #      repetir=1,
    #      x=X,
    #      y=Y,
    #      max_epoca=MAX_EPOCAS,
    #      input=15,
    #      neurons_1=25,
    #      neurons_2=1,
    #      seed=17)
    # log.finish()

    df = pd.read_csv('data/exe_07_teste_rede_03.csv', sep=',', header=None)
    X = df.iloc[0:20, [i for i in range(15)]].values.astype(np.longdouble)
    X = np.insert(X, 0, -1, axis=1)
    Y = df.iloc[0:20, [15]].values.astype(np.longdouble)
    log = CustomLogger()
    
    # w1 = np.array([[-0.89842788, 8.82685116, -2.52482094, 18.26376395, -8.36802671, 5.58476186, -28.99030035, 5.61470914, -6.42843172, 12.98335805, 6.61923656, -0.60039287, -9.10214751, -2.42793517, 0.66250257, 12.52044963], [0.27617089, 0.38198219, 0.2862681, 0.18447938, 1.1190934, 0.92774802, 0.28471835, 0.35170284, 1.0125289, 0.6243768, 0.5782884, -0.15636379, 0.35709917, 0.65182696, 0.20539327, 0.44812531], [0.95106449, 0.65480737, 0.63166387, 0.61937472, 0.53679487, 0.95330884, -0.47623154, 0.84379079, 0.86270816, 0.9537968, 0.37600265, 0.79728132, -0.07132342, 0.39211953, 0.90031272, 0.46746452], [0.55311817, 0.22541609, -0.16332994, 0.77677488, -0.07547622, 0.28663336, -0.21001007, 0.09912975, 0.3641356, 0.04343021, 0.82492944, 0.44030404, 0.0757064, 0.22417114, 0.66462143, 0.41472891], [0.06798989, 0.54027605, 0.61544188, 0.35779748, 0.89960697, 0.5559124, 1.01684055, 0.17637107, 0.12937448, 0.68216541, 0.37594948, 0.09348815, 1.10143397, 0.42525467, 0.76664759, 0.63850307], [1.2395149, 0.3180198, 0.77456748, -0.10681157, 0.39415174, 0.63432961, 0.87354203, 0.73340922, 0.7270805, 0.29667723, -0.08005904, 0.84456586, 0.43253816, 0.53129737, 0.25692118, -0.0274028], [1.14821108, 0.32826681, 0.02534737, 0.41889743, 0.4679699, 0.39913692, 1.21114753, 0.13669207, 0.96335595, 0.5106404, -0.18212523, 0.97976766, 0.54281619, 0.89740987, 0.05414375, -0.04639743], [1.38560112, 0.37196417, 0.00649772, 0.32754244, 0.83544715, 0.70872789, 1.29166573, -0.15233294, 0.07528263, -0.26089873, -0.36452551, 0.24046877, 0.77126316, 0.43403651, 0.3941022, 0.42757782], [0.23609279, 0.10301832, 0.65839363, 0.86708636, 0.73076201, -0.0600857, 1.11390128, 0.06823944, 0.91017408, 0.34337895, 0.73111587, 0.83621484, 0.41044613, 0.05269587, -0.01127086, -0.00359884], [0.17021395, -0.02931631, 0.25357122, 0.55550292, 0.63274559, -0.09584649, 0.26525865, 0.8410322, 0.62281168, 0.07893732, 0.20226614, 0.70691528, 0.18427292, 0.52689155, 0.91962579, 0.73814947], [0.30842572, 0.60353935, 0.52760082, 1.07043738, -0.03868612, 0.10864872, -0.45489379, 0.56790469, 0.47094113, 0.68370044, 0.43448021, 0.93408252, 0.45131818, 0.29799023, 0.54042398, 0.85319324], [0.93244116, 0.30059563, -0.09601422, 0.3848391, 0.40051996, 0.20010354, 0.26063659, 0.07410512, 0.08970597, 0.68353592, 0.33512387, 0.42633067, 0.51811911, 0.78031927, 0.2233877, 0.02214098], [1.34602055, 0.10277002, 0.97072385, -0.32066432, 0.43688199, -0.33486606, 2.51868423, 0.44168702, 1.34215154, -0.27525196, 0.16408928, 0.5946119, 0.652819, 0.48947426, 0.42688645, -0.42928034], [2.04773178, -0.62508362, 0.50471682, -0.24565847, 1.13550251, -0.43972295, 1.58800443, -0.73167959, 0.07148654, 0.22083826, 0.2923778, 0.34962419, -0.39296585, -0.09308351, -0.76995232, 0.15122554], [0.87296554, 1.21264842, 0.37046682, 1.46126789, 0.21312373, 0.88207616, -1.22297146, 0.54856467, 0.44992614, 0.42419033, 0.85285438, 0.09296419, -0.46848055, 0.08261906, 0.45578595, 1.08826475], [1.20987951, 0.78959129, 0.16242649, 0.03249109, 0.25007498, 0.43647429, -0.94204516, 0.30033817, 0.24631749, 0.56763746, 0.11201453, 0.42442505, 0.23082565, 0.60139638, 0.37881487, 0.61632047], [0.55093806, 0.6466183, 0.22865628, 0.43069183, 0.70976197, 0.83813816, 0.90866277, 0.74119694, 0.84881091, 0.20748484, 0.64772877, 0.38313827, 0.61271373, 0.5004772, 0.43118064, 0.23464366], [0.79733327, 0.26063421, 0.95085542, 0.43829608, 0.26909186, 0.33544018, 0.60634022, 0.67296596, 0.89424739, 0.73124014, 0.54873935, 0.26316246, 0.26586664, 0.83875921, 0.43851785, 0.63211191], [0.497964, 0.71578371, 0.55445756, 0.64651087, 0.64684282, 0.43672327, 0.59328976, 0.05577576, 0.44349794, 0.00121652, 0.96495633, 0.18663459, 0.95410534, 0.93085307, 0.72430892, 0.51476777], [0.25048306, 0.25898592, 0.38654355, -0.10747609, 0.7681011, 0.20881894, 1.35798782, 0.77708127, 0.3371555, -0.11784172, 0.10790283, 0.52242774, 0.81631645, 0.25040274, 0.1662155, 0.67325834], [1.00147848, 0.19434045, 0.2136744, -0.40084314, 0.65319142, 0.1453664, 0.58995187, 0.25605706, 0.58384214, 0.28209803, -0.14058164, 0.62385699, 0.06751408, 0.32903989, 0.56707111, -0.35898866], [0.33514666, 0.93080507, 0.86510432, 0.7988058, 0.81741787, 0.99177537, 0.4634135, 0.85827633, 0.05977861, 0.82479134, 0.79053965, 0.61411048, 0.57777198, -0.07519797, 0.7666632, 0.88584371], [0.36024265, 0.28308042, 0.57894297, 0.44305869, 0.70026655, 0.45977504, 0.49463129, 0.99356247, 0.2205897, 0.18568464, 0.94969751, 0.48488049, 0.87624466, 0.55552129, 0.97049485, 0.59118486], [0.50177325, 0.15534753, 0.3566259, 0.06474427, 0.15105016, 0.76191986, -0.00349465, 0.38676106, 0.85663302, 0.72894361, 0.69734456, 0.19894156, 0.43168683, 0.58995591, 0.99516062, 0.23104908], [0.27254602, 0.31450977, 0.71033891, 0.42371715, -0.02782913, 0.49200313, 0.16046861, 0.84052077, 0.27967784, -0.02876511, 0.61044972, 0.76011068, 0.47001896, 0.76050162, 0.54864127, 0.50594824]])
    # w2 = np.array([[1.50979401, 0.06778445, -0.49071872, -1.24454951, -0.40447146, 0.88354597, 0.80208359, 1.09064063, 0.73507421, 0.47342836, -0.68697923, -1.11172985, 1.2964922, 2.68679179, -0.71627299, -1.69776552, -0.5007148, 0.44492986, 0.0141724, 0.31770544, 1.10185661, 0.21938443, -0.79231062, -0.18113915, -0.47438067, -0.5588812]])

    # w1 = np.array([[0.03634228, 2.11362236, 1.94697525, 9.40667464, -2.844967, 2.25747942, -14.90358422, 0.80610115, -1.66658298, 5.84235231, 5.43351206, -1.00363573, -5.38044987, -1.43153896, 0.70947813, 7.66348082], [-0.10198613, -0.38031907, 0.75414191, 0.02941125, 1.77842086, -0.21176666, 0.66153582, -0.53439595, 1.21370853, 0.16120096, 1.39252973, -0.324298, 0.4938199, -0.11781639, 0.80737236, 0.71733177], [0.94035927, 0.10574425, 0.73341158, -0.03368081, 0.69623547, -0.04970484, 1.27762836, 0.16150468, 0.4048342, -0.42663419, -0.09736765, 0.46928834, 0.1811449, 0.47321426, 0.15967741, -0.37487862], [1.3423905, 0.2136192, 0.44890241, -0.6082299, 0.36376378, 0.36302569, 1.77891935, 0.6707873, 0.74009795, -0.02109653, -0.60424124, 0.07761843, 0.5718869, 0.08093045, 0.55531379, -0.50073849], [0.63714474, 0.61604163, 0.21704867, 0.27908483, 1.01772117, 0.40081378, 1.61704842, 0.21149964, 0.07252049, -0.06522313, 0.56227054, 0.81888833, 0.32634666, 0.59199576, 0.82910913, 0.56343887], [0.10955608, 0.3385511, 0.28946632, 0.65959597, 0.72229441, 0.3595236, 0.50720684, 0.54233093, 0.89326326, 0.2818879, 0.00757323, 0.09157957, 0.36082596, 0.48938111, 0.57016827, 0.93921204], [0.17731132, 0.46300204, 0.93485816, 0.46915596, -0.03878601, 0.9448082, 0.44591522, 1.00190693, 0.60192656, 0.99142589, 0.65174182, 0.1360496, 0.63333849, 0.53106298, 0.01979305, 0.83175824], [0.19595067, 0.51952381, 0.70169974, 0.8992749, 0.86263544, 0.11880362, -0.14432115, 0.34139802, 0.38193491, 0.62582168, 1.11849869, 0.52841734, 0.52907044, -0.00214381, 0.42314853, 0.226235], [0.30182766, 0.84902467, 0.36508922, 0.29474389, 0.73312065, 0.49300904, 0.39609354, 0.37861032, 0.65795801, 0.36354407, 0.59137449, 0.53750544, 0.04160512, 0.12670055, 0.72319269, 0.52588599], [0.73859187, 0.65728224, 0.03182842, -0.07150499, 0.74943128, 0.37251622, 1.30448301, 0.74386519, 0.8198148, -0.11730464, 0.81417708, 0.8707358, 0.2586772, 0.41061846, -0.01181069, 0.34559525], [1.10875936, 0.87126762, 0.80507291, -0.15549543, 0.1726906, 0.76170793, 0.96813388, 0.09611975, 0.38046295, 0.36814958, 6.505e-05, 0.04660567, 0.22580931, 0.89563387, 0.73590861, 0.1775733], [1.17238029, -0.26825684, 0.55654988, 0.24326061, 0.22518028, 0.43954443, 1.72070196, 0.04001504, 0.37871691, -0.10972434, 0.29294294, 1.10716541, 0.45850025, 0.28657241, -0.04504339, -0.0567465], [0.87201882, 0.13874635, 0.43948125, 0.01537274, 0.14270378, 0.41215623, 0.86978147, 0.91936663, 0.75604159, 0.36128611, -0.15749205, 0.6613685, 0.04472931, 0.81903984, 0.18740424, 0.37657333], [0.49468617, 0.867501, 0.57733518, 0.76629395, 0.15246049, 0.3275206, 0.0458345, 0.41863139, 0.86220616, 0.65866191, 0.97729537, 0.15833082, 0.02576579, 0.2969313, -0.02218071, 0.94304057], [0.39880518, -0.01903582, 0.71440429, 0.20327081, 0.21753927, 0.49538529, 0.28907784, 0.93816505, 0.07480321, 0.56279387, 0.87560877, 0.84720175, 0.68177881, 0.48711038, 0.76463118, 0.71645107], [0.32505723, 0.2556877, 0.29906861, 1.10721029, 0.85725193, 0.18657121, 0.43589383, 0.47093012, 0.25335863, 0.89650512, 1.07345077, 0.220017, -0.15756208, -0.0472408, 0.75923844, 0.52485902], [0.78248375, 0.2365124, 0.76509647, 0.68749693, -0.19623386, 0.73826658, -0.56725818, 0.47007336, 0.25410563, 1.03721631, 1.05911161, 0.2561606, 0.01481542, 0.75566217, 0.63872531, 0.49976646], [0.91317209, 0.81426548, 0.91398834, -0.03181174, 0.78831272, 0.80376231, -0.04716809, 0.76967089, 0.48929811, 0.48832327, 1.03445383, -0.12796321, 0.27800437, -0.04984411, 0.03950624, 0.62862364], [0.60843778, 0.95247281, 0.25927612, 0.94062073, 0.43590751, 0.68872055, -0.02417852, 0.96214767, 0.21287292, 0.24961043, 0.78291604, 0.86073723, 0.97944978, 0.71899129, 0.40755658, 0.01220998], [0.90901363, 0.82118356, -0.01245552, 0.05104043, 0.77039096, 0.30346653, 0.49260283, 1.00459721, 0.37109417, 0.20165816, 0.11114791, 0.15653609, 0.6406366, 0.14274715, 0.68811336, 0.20964002], [0.53053937, 0.9331981, 0.53765091, 1.22876195, 0.85988976, 0.58405847, 0.40411866, 0.86679493, 0.17997327, 0.61486405, 0.93714832, 0.04024707, 0.38544336, 0.18570061, 0.41212038, 0.37454272], [0.60070262, 0.82869904, 0.96964742, 0.11101863, 0.88276666, 0.7587251, 0.76524574, 0.78353043, 0.31909851, 0.25251295, 0.4970338, 0.04068418, 0.5572666, 0.67028245, 0.19649976, 0.05105298], [1.73987337, 0.02140633, -0.08066211, -0.28424078, 0.64176532, 0.11886876, 1.11394221, -0.07496272, 0.18189204, -0.02604818, -0.37462096, 0.34079514, 1.12472301, 0.94068881, -0.07633989, -0.0487445], [0.86236254, 0.64706879, 0.46797516, 0.64052779, 0.42564117, 0.51649414, 1.05264035, 0.83249082, 0.20345266, 1.04884606, 0.17886065, 0.26069393, 0.71643656, 0.86440101, 0.09780343, 0.95356264], [0.37407463, 0.89576858, 0.71301132, 0.86098583, 0.12989979, 0.69495717, 0.24189185, 0.29738686, 0.17156372, 0.90525858, 0.80618827, 0.69281802, 0.60889936, 0.13920464, 0.97884328, 0.48243938]])
    # w2 = np.array([[0.98733598, 0.05206082, 1.20161108, 1.97118101, 1.36903723, 0.12812456, -0.32441475, -0.78145421, -0.75366922, 0.40456462, 0.77316298, 0.91011858, 1.23770346, -0.46770883, -0.61207677, -0.7791017, -0.98758524, -1.40183479, -0.32604717, 0.12954974, 0.08181696, -0.53942397, 1.318598, 1.30705638, -0.44114007, -0.76619478]])

    w1 = np.array([[-0.31458684, 5.01610222, 0.95856021, 16.36070826, -5.12681926, 4.22191034, -24.65893226, 2.38408137, -4.62878566, 11.60993111, 5.38890385, 1.16360761, -10.29114037, -3.95944358, 0.25000234, 12.26772607], [0.3241406, 0.52123365, 0.61459566, 0.61958898, 1.3532034, 0.18263866, 0.82245899, 0.22016541, 0.54733828, 0.60556725, 1.32501429, 0.40263125, 0.47273473, 0.48577886, 0.54219423, 0.49871849], [0.3243569, 0.78338225, 0.55965622, 0.0292609, 0.7607765, 0.5803739, 0.76037757, 0.54749811, 0.05755577, 0.57788021, 0.49120646, 0.05082959, 0.11825986, 0.31119109, 0.9974638, 0.12737232], [0.11771621, 0.02080662, 0.33531973, 0.83671285, 0.2043716, 0.94058541, 0.94385244, 0.01459998, 0.77082578, 0.71057872, 0.51177304, 0.27223567, 0.63170565, 0.76765709, 0.46642954, 1.00371602], [0.49415146, 0.67555185, 0.42515109, 0.26826717, 0.93485371, 0.69495468, 0.6426971, 0.04226744, 0.89694403, 0.42622755, 0.47294903, 0.1419899, 0.35815411, 0.33326394, 0.14469507, 0.80299582], [0.5415402, 0.89078226, 0.7934963, 0.63667461, 0.06920742, 0.78192238, 0.37946332, 0.47341151, 0.07728293, 0.17162773, 0.35862151, 0.35483852, 0.01463835, 0.23397202, 0.18339109, 0.38872441], [1.26224702, 0.86375489, 0.79866969, 0.9206621, 0.17164315, 0.47135171, -1.35171155, 0.48784297, 0.20536681, 0.87415709, 1.31057322, 0.13920676, -0.16087845, 0.09636141, 0.20718833, 1.31060082], [1.80505351, -0.01492388, 0.34426514, 0.01868188, -0.64961996, 0.84215611, -1.51810392, 0.09122243, -0.71443732, 0.43606358, 0.28871992, 0.07412813, 0.40511423, -0.42159869, 0.06896579, 0.50709526], [0.45821819, 0.72551699, 0.48689996, 0.17278076, 0.40346251, 0.24896627, 0.57796577, 0.89937325, 0.67716361, 0.01602861, 0.4299325, 0.79032589, 0.85778239, 1.07593398, 0.33033768, -0.12277964], [0.88798778, 0.83468631, 0.79222559, 0.51412043, 0.43761897, 0.31469878, 0.48412704, 0.93921869, 0.89323946, 0.90069625, 0.30171833, 0.81602056, 0.27232322, -0.03957592, 0.05210511, 0.68029548], [0.87953691, 0.45306848, 0.2462762, 0.95919172, 0.67117331, 0.84135902, 0.19408124, 0.97460115, 0.724587, 1.0703333, 0.89064396, 0.79991913, 0.03649854, 0.34624667, 0.35288973, 1.06719278], [0.37799998, 0.96142537, 0.66122058, 1.06166293, 0.61461541, 0.59644269, 0.20606713, -0.00536335, 0.69228113, 0.59429873, 0.40416138, 0.62434831, 0.81398419, 0.31909144, 0.38001385, 0.2608239], [0.40207361, 0.79604706, 0.3251751, 0.15866072, 0.29992334, 0.23459923, 0.94737854, 0.9241997, 0.16573345, 0.0096817, 0.4471306, 1.0015026, 0.30204625, 0.83582131, 0.70879462, 0.31060508], [0.76471689, 0.93979051, 0.22778365, 0.74288676, 0.67535306, 0.42015946, 0.79838146, 0.12267046, 0.02334768, 0.57266249, 0.24861978, 0.96785935, 0.81205711, -0.02263568, 0.0143605, 0.40662885], [0.76087068, 0.06757394, 0.28904366, 0.18005612, 0.28134125, 0.52919051, 0.48734726, 0.04379751, 0.58309574, 0.38061274, -0.04120531, 0.02538562, 0.12238232, 0.90085167, 0.69513039, 0.48639424], [0.98672807, 0.20202984, 0.58419663, 0.07882178, 0.81553791, 0.47436343, 0.58231635, 0.72012156, 0.02983728, 0.98476953, 0.63062765, 0.09540968, 0.61615315, 0.74557647, 0.7027333, 0.18811763], [0.83943913, 0.74653754, 0.79005287, 0.02850685, 0.46405769, 0.17189503, 1.46085001, 0.76915932, 0.23502087, -0.07288683, 0.43273659, 0.31687929, 0.56026435, 0.98774192, 0.45549479, 0.3063911], [1.10239825, 0.09466546, 0.13551065, 0.10172037, 0.42105262, 0.79223688, 0.60228255, 0.83054274, 0.56455802, 0.1320102, 0.26606871, 0.65452453, -0.00462337, 0.45537621, 0.529706, 0.79249787], [0.10409256, 0.63620656, 0.69574068, -0.11356503, 0.31379848, 0.4648891, 1.2925764, 0.61687491, 0.75353025, 0.55188243, 0.05055376, 0.55956761, 0.47093572, 0.60935013, 0.44278596, 0.40294211], [1.11760261, 0.48029956, 0.75032296, -0.59573131, 0.3971574, 0.3806072, 1.12683143, 0.05408967, 0.76966272, 0.47710462, -0.27244686, 0.58564051, 0.88719183, 0.06908581, 0.02159704, 0.05865857], [1.22814452, 0.49020254, 0.4706175, -0.5240964, 0.95203421, -0.00594221, 1.16332563, 0.53614661, 0.5526165, 0.55331113, -0.10756356, 0.05107539, 0.09480048, -0.14470207, 0.13645222, -0.22743451], [0.77243747, 0.08998529, 0.60288372, 0.92615941, 0.00631368, 0.77114285, 0.13955978, 0.99664534, 0.40868091, 0.66792355, 0.15375314, 0.04226448, 0.20781608, 0.69519018, 0.58765709, 0.41908142], [1.00276892, 0.9259357, 0.4318899, 0.36546155, 0.21977322, -0.03380893, -0.00387061, 0.82654982, 0.99293797, 0.48534178, 0.485487, 0.71930984, 0.56065594, 0.36315152, 0.07467564, 0.95427572], [0.87010578, 0.17109189, 0.28788474, 0.22494294, 0.57113034, 0.07888944, 0.88353928, 0.01520627, 0.51689558, 0.17878767, 0.67914566, 0.9651538, 0.34613561, 0.68491149, 0.5016341, 0.93335603], [1.38599007, 0.44536634, 0.65349586, -0.4566321, 0.34671395, -0.00509566, 1.23410228, 0.07425791, 0.40883449, 0.09052237, -0.00198297, 0.13999731, 0.37127711, 0.58694371, 0.26739974, -0.3393457]])
    w2 = np.array([[1.2635176, -0.02339529, 0.19432511, 0.01150149, -0.15841191, 0.15257204, -1.45826419, -2.7485207, -0.66381852, 0.34536757, -0.92249238, -0.79505873, 0.33793544, 0.03248213, 0.15742436, 0.21198854, 0.52650786, 0.68293702, 0.16319512, 1.14125888, 1.47325306, 0.47823337, -0.5785495, -0.3933555, 1.0735248, 1.18070377]])
    
    hist, res = teste(X, Y, w1, w2)
    print('Histórico erro: {}'.format(hist))
    print('Predições: {}'.format(res))
    log.finish()