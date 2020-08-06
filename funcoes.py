# Funções úteis 
import pandas as pd
import numpy as np

# Funções essenciais para a organização dos dados
def normalizar_serie(serie):
    minimo = np.min(serie)
    maximo = np.max(serie)
    y_temp = 2*((serie - minimo) / (maximo - minimo)) - 1
    y = y_temp / np.sqrt(len(serie))

    return y

def desnormalizar(serie_atual, serie_real):
    minimo = np.min(serie_real)
    maximo = np.max(serie_real)
    
    serie_temp = serie_atual * np.sqrt(len(serie_real))
    serie_temp2 = (serie_temp + 1)/2
    serie_temp3 = serie_temp2 * (maximo - minimo) + minimo 
    
    return pd.DataFrame(serie_temp3)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in np.arange(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Criando os conjuntos de treinamento, validação e teste
def divisao_dados_temporais(X,y, perc_treino, perc_val = 0):
    tam_treino = int(perc_treino * len(y))
    
    if perc_val > 0:        
        tam_val = int(len(y)*perc_val)
              
        X_treino = X[0:tam_treino,:]
        y_treino = y[0:tam_treino,:]
        
        #print("Particao de Treinamento:", 0, tam_treino)
        
        X_val = X[tam_treino:tam_treino+tam_val,:]
        y_val = y[tam_treino:tam_treino+tam_val,:]
        
        #print("Particao de Validacao:",tam_treino,tam_treino+tam_val)
        
        X_teste = X[(tam_treino+tam_val):-1,:]
        y_teste = y[(tam_treino+tam_val):-1,:]
        
        #print("Particao de Teste:", tam_treino+tam_val, len(y))
        
        return X_treino, y_treino, X_teste, y_teste, X_val, y_val
        
    else:
        
        X_treino = X[0:tam_treino,:]
        y_treino = y[0:tam_treino,:]

        X_teste = X[tam_treino:-1,:]
        y_teste = y[tam_treino:-1,:]

        return X_treino, y_treino, X_teste, y_teste 

# janelamento para cenários dinâmicos
def cenarios_dinamicos(serie, window_size, step_size):
    '''
    Janelamento móvel que envolve selecionar o tamanho da janela (window_size) e o tamanho do passo (step_size).
    
    '''
    w = window_size
    s = step_size
    t = len(serie)
    
    cenarios = []
    
    i_max = int(np.floor((t - w)/s))
    
    for i in range(i_max+1):
        s_temp = serie[(i*s):((i*s)+w)]
        cenarios.append(s_temp)
        
    return cenarios

# Performance metric
def cmf(lista_mse, T):
    '''
    Argumentos:
    lista_mse -> lista contendo os valores do mse (fitness) para cada iteração (t).
    T -> número total de iterações. 
    
    Retorna:
    CMF -> Métrica de desempenho dos algoritmos para todas as iterações
    '''
    lista_mse = np.array(lista_mse)
    return 1/T * lista_mse.sum()

# generalization factor 
def gf(cmf_treino, cmf_teste):
    
    return cmf_teste/cmf_treino