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
    
    for i in range(i_max):
        s_temp = serie[(i*s):((i*s)+w)]
        cenarios.append(s_temp)
        
    return cenarios

# Criando cenários
def cenarios_execucoes(X, y, w, s, f, modelo, qtd_execucoes = 30):
    
    # gerando os cenários dinâmicos
    X_I = cenarios_dinamicos(X, w, s)
    y_I = cenarios_dinamicos(y, w, s)
 
    # calculando a quantidade de iterações
    T = int(f/s*(len(y)-w)+f)
    
    mse_treino = np.zeros((qtd_execucoes, len(y_I) * f))
    mse_teste = np.zeros((qtd_execucoes, len(y_I) * f))

    execucoes = np.arange(qtd_execucoes)

    for execucao in execucoes:
        # inicializando array para 
        mse_treino_lista = np.array([])
        mse_teste_lista = np.array([])
    
        print('Execução: ', execucao)
    
    # Janelamento
        for i in np.arange(len(y_I)):
            print('Janela: ', i)
            ## Divisão em treinamento e teste
            X_treino, y_treino, X_teste, y_teste, X_validacao, y_validacao = divisao_dados_temporais(X_I[i], y_I[i], perc_treino=.56, perc_val = .24)
    
            ### Treinar rede neural com backprop
            # setando parâmetros para comparação
            best_model = 0
            best_mse = np.inf

            # quantidade de neurônios de 2 até 25
            neuronios = np.arange(2, 26)
    
            # grid search 
            for j in neuronios:
                #print('Neurônios: ', j)
                
                # treinar NN para f iterações
                parameters, mse_treino_lista_temp, mse_val_lista_temp, mse_teste_lista_temp = modelo(X_treino.T, y_treino.T, X_validacao.T, y_validacao.T, 
                                                                                                     X_teste.T, y_teste.T, n_h = j, num_iteracoes = f)
                
                # predição na validação
                y_pred_val = predict2(parameters, X_validacao.T)
                mse_validacao = compute_cost2(y_pred_val, y_validacao.T, parameters)
        
                if mse_validacao < best_mse:
                    best_model = parameters 
                    best_mse = mse_validacao
                    #print('Melhor MSE: ', best_mse)
                    qtd_neuronios = j
                        
                    # salvando as listas com os mse de treino e teste
                    best_mse_treino = np.array(mse_treino_lista_temp)
                    best_mse_teste = np.array(mse_teste_lista_temp)

            ## ao final da grid, concatenar as listas de treino e teste    
            mse_treino_lista = np.concatenate((mse_treino_lista, best_mse_treino), axis = None)
            mse_teste_lista = np.concatenate((mse_teste_lista, best_mse_teste), axis = None)
            
        # salvar lista com os mse de treino para todas as iterações
        mse_treino[execucao,:] = mse_treino_lista
        # salvar lista com os mse de teste para todas as iterações
        mse_teste[execucao,:] = mse_teste_lista

    return mse_treino, mse_teste

## Criando avaliação dos resultados
def avaliacao_resultados(mse_treino, mse_teste):
    
    # quantidade de janelas
    ### Garantindo que a quantidade de janelas é igual
    assert mse_treino.shape[1] == mse_teste.shape[1]
    
    qtd_iteracoes = mse_treino.shape[1]
    
    # Calculando CMF
    te = mse_treino.sum(axis=1)/qtd_iteracoes
    ge = mse_teste.sum(axis=1)/qtd_iteracoes

    # calcular a métrica fator de generalização
    gf = ge/te

    # Média e desvio padrão
    te_medio = te.mean()
    te_std = te.std()

    ge_medio = ge.mean()
    ge_std = ge.std()

    gf_medio = gf.mean()
    gf_std = gf.std()

    print('TE medio: ', te_medio)
    print('TE desvio: ', te_std)
    print('GE medio: ', ge_medio)
    print('GE desvio: ', ge_std)
    print('GF medio: ', gf_medio)
    print('GF desvio: ', gf_std)
    
    resultados = {'TE medio': te_medio,
    'TE desvio': te_std,
    'GE medio': ge_medio,
    'GE desvio':ge_std,
    'GF medio':gf_medio,
    'GF desvio':gf_std}
    
    return resultados