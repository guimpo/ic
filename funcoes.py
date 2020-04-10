from scipy.special import expit


def func_logistica(e): return expit(e)


def funcao_degrau(u):
    if(u >= 0):
        return 1
    return 0


def funcao_sinal(u):
    '''
            retorna -1 ou 1
    '''
    if(u >= 0):
        return 1.
    return -1.


def pos_proc(e): return 1 if e >= 0.5 else 0
