#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import random 
import cmath

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 9.0) # set default size of plots

<<<<<<< Updated upstream
###  Modelo para Regressão Com backpropagation
=======

###  Modelo para Regressao Com backpropagation
>>>>>>> Stashed changes

def layer_sizes2(X, Y, n_h=4):
    """
    Argumentos:
    X -- shape do input (quantidade de features, quantidade de exemplos)
    Y -- shape do target (1, quantidade de exemplos)
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters2(n_x, n_h, n_y):
    """
    Argument:
    n_x -- tamanho da camada de entrada
    n_h -- tamanho da camada escondida
    n_y -- tamanho da camada de saída
    
    Retorna:
    params -- dicionário com os parâmetros (pesos) iniciais do modelo:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    

    W1 = np.random.uniform(low = -1/np.sqrt(n_h), high = 1/np.sqrt(n_h), size = (n_h, n_x))
    #W1 = np.random.uniform(low = -1/np.sqrt(n_h), high = 1/np.sqrt(n_h), size = (n_h, n_x)) * 0.01
    #b1 = np.zeros((n_h, 1))
    b1 = np.random.uniform(low = -1/np.sqrt(n_h), high = 1/np.sqrt(n_h),size = (n_h, 1))
    W2 = np.random.uniform(low = -1/np.sqrt(n_y), high = 1/np.sqrt(n_y),size = (n_y, n_h))
    #b2 = np.zeros((n_y, 1))
    b2 = np.random.uniform(low = -1/np.sqrt(n_y), high = 1/np.sqrt(n_y),size = (n_y, 1))
    
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation2(X, parameters):
    """
    Argumentos:
    X -- input de tamanho (n_x, m)
    parametros -- python dicionário contendo os parâmetros (saída da funcao de inicializacao dos parametros)
    
    Retorna:
    A2 -- A saída da funcao sigmoidal ou tangente hiberbólica ou relu
    cache -- dicionário contendo "Z1", "A1", "Z2" e "A2"
    """

    # Recupere cada parâmetro do dicionário parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Implementando a Forward Propagation para calcular A1 tanh e A2 linear
    Z1 = np.dot(W1,X) + b1
    A1 = Z1 # linear
    Z2 = np.dot(W2, A1) + b2
    A2 = 1.7159*np.tanh(2/3 * Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


def compute_cost2(A2, Y, parameters):
    """
    Computa o custo dado os argumentos
    
    Arguments:
    A2 -- Saída linear da segunda ativacao de shape (1, qtd de exemplos)
    Y -- Valor verdadeiro do rótulo de shape (1, qtd de exemplos)
    parameters -- dicionário contendo os parâmetros W1, b1, W2 and b2
    
    Retorna:
    cost
    
    """
    
    m = Y.shape[1] # quantidade de exemplos

    # Computa o custo (cost)
    err = A2 - Y
    cost = 1/m * np.sum(err**2)
    
    cost = float(np.squeeze(cost))  # garanta que o custo tem a dimensao esperada
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation2(parameters, cache, X, Y):
    """
    Implementa a retropropagacao 
    
    Argumentos:
    parameters -- dicionário contendo os parâmetros
    cache -- dicionário contendo "Z1", "A1", "Z2" and "A2".
    X -- input de shape (qtd de features, qtd de examplos)
    Y -- valor verdadeiro do rótulo de shape (1, qtd de examplos)
    
    Retorna:
    grads -- dicionário contendo os gradientes em relacao aos diferentes parâmetros
    """
    m = X.shape[1]
    
    # Primeiro, recuperamos W1 e W2 do dicinário "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
        
    # Recuperamos também A1 e A2 do dicionário "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    Z2 = cache['Z2']
    
    # Retropropagacao: calcula-se dW1, db1, dW2, db2.
    dZ2 = (A2 - Y)* (2/3/1.7159 - np.tanh(2/3*Z2)**2)
    #dZ2 = (A2 - Y)*(1.14393 - (A2**2)/1.5)
    dW2 = 1/m * np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims=True)
    #dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1, 2))
    #dZ1 = np.dot(W2.T, dZ2) * A1
    dZ1 = np.dot(W2.T, dZ2)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims=True)
    

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters2(parameters, grads, learning_rate = 1.2):
    """
    Atualiza os parâmetros utilizando o gradient descendente 
    
    Argumentos:
    parameters -- dicionário contendo os parâmetros
    grads -- dicionário contendo os gradientes
    
    Retorna:
    parameters -- dicionário contendo os parâmetros atualizados
    """
    # Recupera-se cada parâmetro do dicionário "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Recupera-se cada gradiente do dicionário "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    # Regra de atualizacao para cada parâmetro
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model2(X, Y, n_h, num_iteracoes, perc_treino, perc_val,print_cost = False):
#def nn_model2(X, Y, X_val, Y_val, X_test, Y_test, n_h, num_iteracoes, print_cost=False):
    """
    Argumentos:
    X -- dataset de shape (2, qtd de examplos)
    Y -- labels de shape (1, qtd de examplos)
    n_h -- tamanho da camada escondida
    num_iteracoes
    print_cost -- se True, mostra o custo a cada 1000 iteracões
    
    Retorna:
    parameters -- parâmetros aprendidos pelo modelo. Eles podem ser utilizados para fazer previsões (predict).
    """
    
    treino_mse = []
    val_mse = []
    teste_mse = []
    
    n_x = layer_sizes2(X[0].T, Y[0].T)[0]
    n_y = layer_sizes2(X[0].T, Y[0].T)[2]

    # Inicializacao dos parâmetros
    parameters = initialize_parameters2(n_x, n_h, n_y)
    
    for janela in np.arange(len(Y)):
        X_treino, Y_treino, X_teste, Y_teste, X_val, Y_val = divisao_dados_temporais(X[janela], Y[janela], perc_treino, perc_val)
        
        X_tv = np.hstack((X_treino.T, X_val.T))
        Y_tv = np.hstack((Y_treino.T, Y_val.T))
        
        # Gradiente descendente (loop)
        for i in range(0, num_iteracoes):
        
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = forward_propagation2(X_treino.T, parameters)
        
            # Funcao de custo. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = compute_cost2(A2, Y_treino.T, parameters)
 
            # Retropropagacao (Backpropagation). Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = backward_propagation2(parameters, cache, X_treino.T, Y_treino.T)
 
            # Atualizacao dos parâmetros pelo gradiente descendente. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = update_parameters2(parameters, grads, learning_rate=1.2)
        
            # computando mse de treino, teste e validacao
            ## validacao
            A2_val = predict2(parameters, X_val.T)
            mse_val_temp = compute_cost2(A2_val, Y_val.T, parameters)
            val_mse.append(mse_val_temp)
        
            ## treino e val
            A2_tv = predict2(parameters, X_tv)
            mse_tv_temp = compute_cost2(A2_tv, Y_tv, parameters)
            treino_mse.append(mse_tv_temp)
        
            ## teste
            A2_test = predict2(parameters, X_teste.T)
            mse_test_temp = compute_cost2(A2_test, Y_teste.T, parameters)
            teste_mse.append(mse_test_temp)
        
    return parameters, treino_mse, val_mse, teste_mse

def predict2(parameters, X):
    """
    Utiliza os parâmetros aprendidos para prever o valor da saída para cada exemplo X 
    
    Argumentos:
    parameters -- dicionário contendo os parâmetros
    X -- input de tamanho (n_x, m)
    
    Retorna
    predictions -- vetor de valores previstos do modelo treinado
    """
    
    A2, cache = forward_propagation2(X, parameters)
    predictions = A2
    
    return predictions


### Estruturando os dados de dicionário para numpy array e de numpy array para dicionário
def parameter_dim_tot(parameter):
    '''
    Argumentos:
    parameter - array de parâmetros

    Retorna:
    dim_tot - dimensao total dos parâmetris 
    '''
    dim_tot = np.array(parameter.shape).prod()

    return dim_tot

def parameter_reshape_coluna(parameter):
    '''
    Argumentos:
    parameter - array de parâmetros

    Retorna:
    parameter_reshaped - array coluna dos parâmetros
    '''
    param_dim_tot = parameter_dim_tot(parameter)
    parameter_reshaped = parameter.reshape(1, param_dim_tot)

    return parameter_reshaped

def parameters_stack(parameters):
    '''
    Argumentos: 
    parameters - lista com os parâmetros em array 

    Retorna:
    parametros_stack - array coluna com parâmetros empilhados
    '''
    params_list = []
    param_temp = 0
    
    for param in parameters:
        param_temp = parameter_reshape_coluna(param)
        params_list.append(param_temp)
    
    params_stack = np.concatenate(tuple(params_list), axis = 1)

    return params_stack

# Unstack os parâmetros com base na dimensao dos atributos (matrizes de pesos)
def parameters_unstack(parameters_stack, atributos_dim):
    '''
    Argumentos:
    parameters_stack - array dos parâmetros no formato empilhado por colunas para trabalhar no PSO
    atributos_dim - lista com dimensao total dos atributos 

    Retorna:
    params - lista com parâmetros no formato de lista
    '''
    params = []
    i = atributos_dim[0]
    params.append(parameters_stack[:, :i])

    for dim in atributos_dim[1:]:
        params.append(parameters_stack[:, i:i+dim])
        i += dim

    return params

# Reshape para o formato do dicionário (parameters)
def parameters_reshape_dictionary(parameters_dict, parameters_unstacked):
    '''
    Argumentos:
    parameters_dict - dicionário com parâmetros
    parameters_unstacked - parâmetros no formato array ts: 'X_val', 'Y_val', 'X_test', 'Y_test'

    retorna:
    parameters_reshaped - lista com os parâmetros formatados para o dicionário 'parameters'
    '''
    w1_shape = parameters_dict['W1'].shape
    b1_shape = parameters_dict['b1'].shape
    w2_shape = parameters_dict['W2'].shape
    b2_shape = parameters_dict['b2'].shape

    parameters_reshaped = parameters_dict.copy()

    w1_reshaped = parameters_unstacked[0].reshape(w1_shape)
    b1_reshaped = parameters_unstacked[1].reshape(b1_shape)
    w2_reshaped = parameters_unstacked[2].reshape(w2_shape)
    b2_reshaped = parameters_unstacked[3].reshape(b2_shape)

    parameters_reshaped['W1'] = w1_reshaped
    parameters_reshaped['b1'] = b1_reshaped
    parameters_reshaped['W2'] = w2_reshaped
    parameters_reshaped['b2'] = b2_reshaped

    return parameters_reshaped


# In[215]:


#### PSO para otimizar todos os parâmetros de uma só vez 
    
def PSO_todos(X,parameters_stacked, best_cost, fun, A2, Y, parameters, qtd_particulas, atributos_dim, min_i, max_i, 
                max_epoch, perc_treino, perc_val, w_in=0.7, w_fim = 0.2, c1=1.496, c2=1.496):
    '''
        Funcao do Algoritmo SWARM PSO. 
        Inputs:
        - fun_opt: Funcao de fitness a ser otimizada
        - qtd_particulas: Quantidade de partículas
        - atributos_dim: Dimensao do Vetor de atributos 
        - min: intervalo inferior do domínio da funcao  
        - max: intervalo superior do domínio da funcao
        - w: inércia 
        - c1: influência do pbest (termo cognitivo)
        - c2: influência do gbest (termo do aprendizado social)
    '''
    
    treino_mse = []
    val_mse = []
    teste_mse = []
    
    def weight_decay(w_in, w_fim, iter, iter_max):
        
        return w_in + w_fim * (1 - (iter/iter_max))

    
    atributos_dim_sum = sum(atributos_dim)

    # inicializar as partículas em posicões aleatórias
    particulas = np.random.uniform(low = min_i, high = max_i, size = (qtd_particulas, atributos_dim_sum))

    # inicializar a velocidade
    velocidade = np.zeros((qtd_particulas, atributos_dim_sum))

    # inicializar o pbest em zero
    pbest = np.zeros((qtd_particulas,atributos_dim_sum))

    gbest_value = best_cost
    #print('Custo gbest inicio PSO = ', gbest_value)

    gbest = 0
    #particulas[gbest,:] = parameters_stacked
    
    parameters_gbest_dict = parameters.copy()
    parameters_dict = parameters.copy()

    # Extrair a posicao do gbest 
    for z in np.arange(qtd_particulas):
        parameters_temp = particulas[[z],:]
        parameters_temp_unstacked = parameters_unstack(parameters_temp, atributos_dim)
        parameters_temp_dict = parameters_reshape_dictionary(parameters_dict, parameters_temp_unstacked)
        A2 = predict2(parameters_temp_dict, X[0].T)
        new_value = fun(A2, Y[0], parameters_temp_dict)

        if new_value < gbest_value:
            gbest_value = new_value
            gbest = z
            parameters_gbest_dict = parameters_temp_dict

    
    for janela in np.arange(len(Y)):
        
        X_treino, Y_treino, X_teste, Y_teste, X_val, Y_val = divisao_dados_temporais(X[janela], Y[janela], perc_treino, perc_val)
        
        X_tv = np.hstack((X_treino.T, X_val.T))
        Y_tv = np.hstack((Y_treino.T, Y_val.T))
        
        for k in np.arange(max_epoch):
        #print('Iteracao: ', k)
        # Atualizacao do decaimento do peso
            w = weight_decay(w_in, w_fim,k, max_epoch)                
        
        # Iterar para atualizar o pbest e gbest para cada partrícula
            for j in np.arange(qtd_particulas):
        
            # transformando as partículas no formato de dicionário
                parameters_temp = particulas[[j],:]
                parameters_temp_unstacked = parameters_unstack(parameters_temp, atributos_dim)
                parameters_temp_dict = parameters_reshape_dictionary(parameters_dict, parameters_temp_unstacked)

                parameters_pbest_temp = pbest[[j],:]
                parameters_pbest_temp_unstacked = parameters_unstack(parameters_pbest_temp, atributos_dim)
                parameters_pbest_dict = parameters_reshape_dictionary(parameters_dict, parameters_temp_unstacked)

                A2_part = predict2(parameters_temp_dict, X_treino.T)
                A2_pbest = predict2(parameters_pbest_dict, X_treino.T)
            
                # pbest
                if fun(A2_part, Y_treino.T, parameters_temp_dict) < fun(A2_pbest, Y_treino.T, parameters_pbest_dict):
                    pbest[j,:] = particulas[j,:]

            # gbest
                if fun(A2_part, Y_treino.T, parameters_temp_dict) < gbest_value:
                    if np.abs(fun(A2_part, Y_treino.T, parameters_temp_dict) - gbest_value) < 0.00001:
                    
                        gbest_value = fun(A2_part, Y_treino.T, parameters_temp_dict)
                        gbest = j
                        parameters_gbest_dict = parameters_temp_dict
                        break
        
                    gbest = j
                    gbest_value = fun(A2_part, Y_treino.T, parameters_temp_dict)
                    parameters_gbest_dict = parameters_temp_dict
                                      
         # Iteracao para atualizar as posicões das partículas
            for i in np.arange(qtd_particulas):
                r1, r2 = np.random.rand(), np.random.rand()
                velocidade[i, :] = w * velocidade[i, :] + c1 * r1 * (pbest[i, :] - particulas[i, :]) + c2 * r2 * (particulas[gbest, :] - particulas[i, :])
                # Atualizar partículas
                particulas[i, :] = particulas[i, :] + velocidade[i, :]

            # lidar com limites das partículas
                for dim in np.arange(atributos_dim_sum):
                    if particulas[i, dim] < min_i:
                        particulas[i, dim] = min_i
                    elif particulas[i, dim] > max_i:
                        particulas[i, dim] = max_i
        

            # treino e validacao mse
            A2_gbest_tv = predict2(parameters_gbest_dict, X_tv)
            mse_tv = fun(A2_gbest_tv, Y_tv, parameters_gbest_dict)
            treino_mse.append(mse_tv)
        
            # validacao mse
            A2_gbest_v = predict2(parameters_gbest_dict, X_val.T)
            mse_v = fun(A2_gbest_v, Y_val.T, parameters_gbest_dict)
            val_mse.append(mse_v)
        
            # teste
            A2_gbest_t = predict2(parameters_gbest_dict, X_teste.T)
            mse_t = fun(A2_gbest_t, Y_teste.T, parameters_gbest_dict)
            teste_mse.append(mse_t)
        
    return parameters_gbest_dict, treino_mse, val_mse, teste_mse

def update_parameters_pso_todos(X, parameters, best_cost, compute_cost2, A2, Y, perc_treino, perc_val, num_iteracoes):
    '''
    Argumentos:
    parameters - dicionário contendo os parâmetros do modelo
    compute_cost2 - funcao a ser minimizada, neste caso a funcao de custo
    A2 - previsao feita pelo modelo
    Y - rótulo 

    Retorna:
    parameters - parâmetros atualizados a partir do PSO
    '''

    # Extrair os parâmetros do dicionário para calcular a dimensao total e para criar o array colunas
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Extrair a dimensao total 
    W1_dim = np.array(W1.shape).prod()
    b1_dim = np.array(b1.shape).prod()
    W2_dim = np.array(W2.shape).prod()
    b2_dim = np.array(b2.shape).prod()

    # lista com parâmetros
    parametros = [W1, b1, W2, b2]
    # parâmetros no formato array colunas
    parameters_stacked = parameters_stack(parametros)

    atributos_dim = [W1_dim, b1_dim, W2_dim, b2_dim]

    qtd_particulas_dim = (W1.shape[1] + 1)* W1.shape[0] + (W1.shape[0] + 1)*W2.shape[0]

    parameters_pso, treino_mse, val_mse, teste_mse = PSO_todos(X, parameters_stacked, 
                               best_cost,compute_cost2, A2, Y, parameters, qtd_particulas = qtd_particulas_dim, 
                               atributos_dim = atributos_dim, min_i = -1, max_i = 1, max_epoch = num_iteracoes, perc_treino = perc_treino, perc_val=perc_val)

    return parameters_pso, treino_mse, val_mse, teste_mse

def nn_model_pso_todos(X, Y, n_h, num_iteracoes, perc_treino, perc_val, print_cost=False):
    """
    Argumentos:
    X -- dataset de shape (2, qtd de exemplos)
    Y -- labels de shape (1, qtd de exemplos)
    n_h -- tamanho da camada escondida
    num_iteracoes
    print_cost -- se True, mostra o custo a cada 1000 iteracões
    
    Retorna:
    parameters -- parâmetros aprendidos pelo pso. Eles podem ser utilizados para fazer previsões (predict).
    """
    
    n_x = layer_sizes2(X[0].T, Y[0].T)[0]
    n_y = layer_sizes2(X[0].T, Y[0].T)[2]
    
    # Inicializacao dos parâmetros
    parameters = initialize_parameters2(n_x, n_h, n_y)
    
    A2, _ = forward_propagation2(X[0].T, parameters)

    best_cost = compute_cost2(A2, Y[0].T, parameters)
    
    # Atualizacao dos parâmetros pelo gradiente descendente. Inputs: "parameters, compute_cost2, A2, Y". Outputs: "parameters".
    parameters, treino_mse, val_mse, teste_mse = update_parameters_pso_todos(X=X, parameters=parameters, best_cost=best_cost,compute_cost2=compute_cost2, 
                                                                             A2=A2, Y=Y, num_iteracoes=num_iteracoes, perc_treino=perc_treino, perc_val=perc_val)

  
    return parameters, treino_mse, val_mse, teste_mse
    

#### CQSO para otimizar todos os parâmetros de uma só vez 

def CQSO_todos(X, parameters_stacked, best_cost, fun, A2, Y, parameters, qtd_particulas, atributos_dim, min_i, max_i, max_epoch, perc_treino, 
              perc_val, w_in=0.7, w_fim=0.2, c1 = 1.496, c2 = 1.496, neutral_p=25, rcloud=0.2):
    '''
    CQSO algorithm to optimize neural networks weights and biases 
    '''
    # Initialization
    def weight_decay(w_in, w_fim, iter, iter_max):
        return w_in + w_fim * (1 - (iter/iter_max))
        
    atributos_dim_sum = sum(atributos_dim)
        
    n = atributos_dim_sum
    
    a=[]
    
    for i in range(2,n+1):
        if(n%i==0):
            a.append(i)
    a.sort()
    n_sub_swarms = a[0]
    
    # Divide the dimensions per subswarm
    num, div = atributos_dim_sum, n_sub_swarms
    
    dimensions_list = [num // div + (1 if x < num % div else 0)  for x in range (div)]

    if not atributos_dim_sum % n_sub_swarms == 0:
        print("We can't continue, the number of dimensions isn't divisible by the number of subswarms")
        return False

    # Initialization
    context_vector = np.empty(n_sub_swarms, dtype=object)

    ## Create a multiswarm and his velocities
    multi_swarm_vector = np.empty((n_sub_swarms,qtd_particulas), dtype=object)
    velocity_vector = np.empty((n_sub_swarms,qtd_particulas), dtype=object)
    
    ### Change None values for random numbers
    for i_subswarm in range(n_sub_swarms):
        context_vector[i_subswarm] = np.random.uniform(min_i,max_i,(dimensions_list[i_subswarm]))
        for i_particle in range(qtd_particulas):
            multi_swarm_vector[i_subswarm][i_particle] = np.random.uniform(min_i,max_i,(dimensions_list[i_subswarm]))
            velocity_vector[i_subswarm][i_particle] = np.zeros(dimensions_list[i_subswarm])


    ## Create fitness for pbest and gbest
    gbest = np.copy(multi_swarm_vector[0][0])
    pbest = np.copy(multi_swarm_vector[0][0])
    
    sub_swarm_pbest = np.copy(context_vector)
    parameters_dict = parameters.copy()
    
    
    ## transformando lista em dicionário para calcular fitness
    context_vector_unstacked = parameters_unstack(np.concatenate(context_vector).reshape(1,atributos_dim_sum), atributos_dim)
    context_vector_dict = parameters_reshape_dictionary(parameters_dict, context_vector_unstacked)
    A2 = predict2(context_vector_dict, X[0].T)
    pbest_value = fun(A2, Y[0].T, context_vector_dict)
    gbest_value = pbest_value
    
    parameters_gbest_dict = context_vector_dict
     
    
    result_list = []
    
    treino_mse = []
    val_mse = []
    teste_mse = []

    for janela in np.arange(len(Y)):
    
        X_treino, Y_treino, X_teste, Y_teste, X_val, Y_val = divisao_dados_temporais(X[janela], Y[janela], perc_treino, perc_val)
        
        X_tv = np.hstack((X_treino.T, X_val.T))
        Y_tv = np.hstack((Y_treino.T, Y_val.T))
    
        iteration = 0
        while iteration < max_epoch:
        
            w = weight_decay(w_in, w_fim,iteration, max_epoch)       
            
        # Iterations
        # for sub_swarm in multi_swarm_vector:
        
            for i_sub_swarm in range(n_sub_swarms):
            # for particle in sub_swarm:
            
                for i_particle in range(qtd_particulas):
                # Calculate the fitness
                # Vamos calcular o fitness da particula dentro do vetor contextos
                
                    context_copy = np.copy(context_vector)
                    particle = multi_swarm_vector[i_sub_swarm][i_particle]
                    context_copy[i_sub_swarm] = particle
                    
                    parameters_temp_unstacked = parameters_unstack(np.concatenate(context_copy).reshape(1,atributos_dim_sum), atributos_dim)
                    parameters_temp_dict = parameters_reshape_dictionary(parameters_dict, parameters_temp_unstacked)
                    
                    A2_part = predict2(parameters_temp_dict, X_treino.T)
                    fitness_candidate = fun(A2_part, Y_treino.T, parameters_temp_dict)
                
                    if fitness_candidate < pbest_value:
                    # Se o fitness da nova particula for melhor ela vira o pbest
                        pbest = np.copy(multi_swarm_vector[i_sub_swarm][i_particle])
                        pbest_value = fitness_candidate
                        sub_swarm_pbest = np.copy(context_copy)
                        # Feito o pbest devemos atualizar as posicoes das particulas
                
                    if i_particle <= (neutral_p - 1):
                    # Atualiza como PSO vanilla
                        new_velocity = (w * velocity_vector[i_sub_swarm][i_particle]) +                         ((c1 * random.random()) * (pbest - multi_swarm_vector[i_sub_swarm][i_particle])) +                         ((c2 * random.random()) * (gbest - multi_swarm_vector[i_sub_swarm][i_particle]))
                        new_position = new_velocity + multi_swarm_vector[i_sub_swarm][i_particle]
                
                    else:
                    # Atualiza como QSO
                        dist = cmath.sqrt(sum((multi_swarm_vector[i_sub_swarm][i_particle] - gbest)**2))
                        normal = np.random.normal(0, 1, dimensions_list[i_subswarm])
                        uniform = random.choice(np.random.uniform(0, 1, dimensions_list[i_subswarm]))
                        left_size_form = rcloud * normal
                
                        if dist == 0:
                            break
                    
                        right_size_form = (uniform ** (1/dimensions_list[i_subswarm]))/ dist
                        new_position = left_size_form * right_size_form
                
                # Check if the positions is var_min<x<var_max
                    for value in new_position:
                        index = list(new_position).index(value)
                        new_position[index] = np.max([min_i, value])
                        new_position[index] = np.min([max_i, new_position[index]])
                    multi_swarm_vector[i_sub_swarm][i_particle] = new_position
      
                # Visto todas as particulas do subswarm eu comparo o gbest
                if pbest_value < gbest_value:
                    gbest = np.copy(pbest)
                    gbest_value = pbest_value
                    context_vector = np.copy(sub_swarm_pbest)
                    
                    parameters_gbest_unstacked = parameters_unstack(np.concatenate(context_vector).reshape(1,atributos_dim_sum), atributos_dim)
                    parameters_gbest_dict = parameters_reshape_dictionary(parameters_dict, parameters_gbest_unstacked)
                    
            result_list.append(gbest_value)
            iteration += 1
        
            # treino e validacao mse
            A2_gbest_tv = predict2(parameters_gbest_dict, X_tv)
            mse_tv = fun(A2_gbest_tv, Y_tv, parameters_gbest_dict)
            treino_mse.append(mse_tv)
        
            # validacao mse
            A2_gbest_v = predict2(parameters_gbest_dict, X_val.T)
            mse_v = fun(A2_gbest_v, Y_val.T, parameters_gbest_dict)
            val_mse.append(mse_v)
        
            # teste
            A2_gbest_t = predict2(parameters_gbest_dict, X_teste.T)
            mse_t = fun(A2_gbest_t, Y_teste.T, parameters_gbest_dict)
            teste_mse.append(mse_t)
    
    
    return parameters_gbest_dict, treino_mse, val_mse, teste_mse
    
    
def update_parameters_cqso_todos(X, parameters, best_cost, compute_cost2, A2, Y, perc_treino, perc_val, num_iteracoes):
    '''
    Argumentos:
    parameters - dicionário contendo os parâmetros do modelo
    compute_cost2 - funcao a ser minimizada, neste caso a funcao de custo
    A2 - previsao feita pelo modelo
    Y - rótulo 

    Retorna:
    parameters - parâmetros atualizados a partir do PSO
    '''

    # Extrair os parâmetros do dicionário para calcular a dimensao total e para criar o array colunas
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Extrair a dimensao total 
    W1_dim = np.array(W1.shape).prod()
    b1_dim = np.array(b1.shape).prod()
    W2_dim = np.array(W2.shape).prod()
    b2_dim = np.array(b2.shape).prod()

    # lista com parâmetros
    parametros = [W1, b1, W2, b2]
    # parâmetros no formato array colunas
    parameters_stacked = parameters_stack(parametros)

    atributos_dim = [W1_dim, b1_dim, W2_dim, b2_dim]

    qtd_particulas_dim = (W1.shape[1] + 1)* W1.shape[0] + (W1.shape[0] + 1)*W2.shape[0]

    parameters_cqso, treino_mse, val_mse, teste_mse = CQSO_todos(X, parameters_stacked, 
                               best_cost, compute_cost2, A2, Y, parameters=parameters, qtd_particulas = qtd_particulas_dim, 
                               atributos_dim=atributos_dim, min_i = -1, max_i = 1, max_epoch = num_iteracoes, perc_treino=perc_treino, perc_val=perc_val)

    return parameters_cqso, treino_mse, val_mse, teste_mse

def nn_model_cqso_todos(X, Y, n_h, num_iteracoes, perc_treino, perc_val, print_cost=False):
    """
    Argumentos:
    X -- dataset de shape (2, qtd de exemplos)
    Y -- labels de shape (1, qtd de exemplos)
    n_h -- tamanho da camada escondida
    num_iteracoes
    print_cost -- se True, mostra o custo a cada 1000 iteracões
    
    Retorna:
    parameters -- parâmetros aprendidos pelo pso. Eles podem ser utilizados para fazer previsões (predict).
    """
    
    n_x = layer_sizes2(X[0].T, Y[0].T)[0]
    n_y = layer_sizes2(X[0].T, Y[0].T)[2]
    
    # Inicializacao dos parâmetros
    parameters = initialize_parameters2(n_x, n_h, n_y)
    
    A2, _ = forward_propagation2(X[0].T, parameters)

    best_cost = compute_cost2(A2, Y[0].T, parameters)
    
    # Atualizacao dos parâmetros pelo cqsos. Inputs: "parameters, compute_cost2, A2, Y". Outputs: "parameters".
    parameters, treino_mse, val_mse, teste_mse = update_parameters_cqso_todos(X=X, parameters=parameters, best_cost=best_cost,compute_cost2=compute_cost2, 
                                                                             A2=A2, Y=Y, num_iteracoes=num_iteracoes, perc_treino=perc_treino, perc_val=perc_val)
    
    return parameters, treino_mse, val_mse, teste_mse


### Experimento com as as séries do artigo 

## Importar funcões para pre-processar os dados
from funcoes import split_sequence, divisao_dados_temporais, normalizar_serie, desnormalizar

import time


# ## 1. Sunspot annual measure time series (SAM)
# 
# * 289 obs
# * 1770 to 1988
# 
# Série anual inputs: 10


## Sunspot
sunspot = pd.read_csv('dados/sunspot.csv')
sunspot = sunspot['valor']
sunspot.head()


sunspot_norm = normalizar_serie(sunspot)
X, y = split_sequence(sunspot_norm.values, 10, 1)


# ### Cenário I - Sunspot (SAM)
# 
# Testando Cenário I
# 
# * w = 60
# * s = 10
# * f = 50


# janelamento para cenários dinâmicos
def cenarios_dinamicos(serie, window_size, step_size):
    '''
    Janelamento móvel que envolve selecionar o tamanho da janela (window_size) e o tamanho do passo (step_size).
    
    '''
    w = window_size
    s = step_size
    t = len(serie)
    
    cenarios = []
    
    i_max = int((t - w)/s)

    for i in range(i_max):
        s_temp = serie[(i*s):((i*s)+w)]
        cenarios.append(s_temp)
        
    return cenarios



# Criando cenários
def cenarios_execucoes(X, y, w, s, f, modelo, perc_treino, perc_val,qtd_execucoes = 30):
    
    # gerando os cenários dinâmicos
    X_I = cenarios_dinamicos(X, w, s)
    y_I = cenarios_dinamicos(y, w, s)
 
    # calculando a quantidade de iteracões
    T = int(f/s*(len(y)-w)+f)
    
    neuronios = np.arange(2, 26)
    
    mse_treino = np.zeros((qtd_execucoes, len(neuronios),len(y_I) * f))
    mse_val = np.zeros((qtd_execucoes, len(neuronios), len(y_I) * f))
    mse_teste = np.zeros((qtd_execucoes, len(neuronios),len(y_I) * f))

    execucoes = np.arange(qtd_execucoes)

    for execucao in execucoes:
        print('Execucao: ', execucao)

        # Neuronios
        for j,z in zip(neuronios, np.arange(len(neuronios))):
            
            parameters, mse_treino_lista_temp, mse_val_lista_temp, mse_teste_lista_temp = modelo(X_I, y_I, n_h = j, 
                                                                                                 num_iteracoes = f, 
                                                                                                 perc_treino=perc_treino, 
                                                                                                 perc_val=perc_val)

            # salvar lista com os mse de treino para todas as iteracões
            mse_treino[execucao, z,:] = np.array(mse_treino_lista_temp)
            # salvar lista com os mse de validacao para todas as iteracoes
            mse_val[execucao, z,:] = np.array(mse_val_lista_temp)
            # salvar lista com os mse de teste para todas as iteracões
            mse_teste[execucao, z,:] = np.array(mse_teste_lista_temp)

    return mse_treino, mse_val, mse_teste


def avaliacao_resultados(mse_treino_cenarios, mse_val_cenarios, mse_teste_cenarios, f, quantidade_janelas, execucoes):
    
    mse_treino = np.zeros((execucoes, quantidade_janelas*f))
    mse_teste = np.zeros((execucoes, quantidade_janelas*f))

    for ex in np.arange(execucoes):
        id_neuronios = [np.nanargmin(mse_val_cenarios[ex,:,f*janela]) for janela in range(quantidade_janelas)]
        for jan in np.arange(quantidade_janelas):
            mse_treino[ex, f*jan:f*jan+f] = mse_treino_cenarios[ex, id_neuronios[jan], f*jan:f*jan+f]
            mse_teste[ex, f*jan:f*jan+f] = mse_teste_cenarios[ex, id_neuronios[jan], f*jan:f*jan+f]
    
    qtd_iteracoes = mse_treino.shape[1]
    # Calculando CMF
    te = mse_treino.sum(axis=1)/qtd_iteracoes
    ge = mse_teste.sum(axis=1)/qtd_iteracoes

    # calcular a métrica fator de generalizacao
    gf = ge/te

    # Média e desvio padrao
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
    
    resultados = {'TE medio': [te_medio],
    'TE desvio': [te_std],
    'GE medio': [ge_medio],
    'GE desvio':[ge_std],
    'GF medio':[gf_medio],
    'GF desvio':[gf_std]}
    
    return resultados, mse_treino, mse_teste


# Quantidade total de iteracões para o primeiro cenário

# Seed
np.random.seed(3)

w = 60 # tamanho da janela
s = 10 # tamanho do passo
f = 50 # quantidade de iteracões para a janela
T = int(f/s*(len(y)-w)+f)

quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
# #backprop
print('BACKPROP')
sam_mse_treino_1_backprop, sam_mse_val_1_backprop, sam_mse_teste_1_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
sam_resultados_1_backprop, sam_resultados_mse_treino_1_backprop, sam_resultados_mse_teste_1_backprop = avaliacao_resultados(sam_mse_treino_1_backprop, 
                                                                                                                             sam_mse_val_1_backprop, 
                                                                                                                             sam_mse_teste_1_backprop, 
                                                                                                                             f, quantidade_janelas, 
                                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(sam_resultados_1_backprop).to_csv('resultados/sam_resultados_1_backprop.csv')
pd.DataFrame(sam_resultados_mse_treino_1_backprop).to_csv('resultados/sam_resultados_mse_treino_1_backprop.csv')
pd.DataFrame(sam_resultados_mse_teste_1_backprop).to_csv('resultados/sam_resultados_mse_teste_1_backprop.csv')

print('PSO')
# # pso
tic = time.time()
sam_mse_treino_1_pso, sam_mse_val_1_pso, sam_mse_teste_1_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                   perc_treino=0.54, perc_val=0.24)
sam_resultados_1_pso, sam_resultados_mse_treino_1_pso, sam_resultados_mse_teste_1_pso = avaliacao_resultados(sam_mse_treino_1_pso, 
                                                                                                  sam_mse_val_1_pso,sam_mse_teste_1_pso, 
                                                                                                              f, quantidade_janelas, 
                                                                                                              execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(sam_resultados_1_pso).to_csv('resultados/sam_resultados_1_pso.csv')
pd.DataFrame(sam_resultados_mse_treino_1_pso).to_csv('resultados/sam_resultados_mse_treino_1_pso.csv')
pd.DataFrame(sam_resultados_mse_teste_1_pso).to_csv('resultados/sam_resultados_mse_teste_1_pso.csv')

print('CQSO')
# cqso
tic = time.time()
sam_mse_treino_1_cqso, sam_mse_val_1_cqso, sam_mse_teste_1_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sam_resultados_1_cqso, sam_resultados_mse_treino_1_cqso, sam_resultados_mse_teste_1_cqso = avaliacao_resultados(sam_mse_treino_1_cqso, 
                                                                                                             sam_mse_val_1_cqso,
                                                                                                             sam_mse_teste_1_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(sam_resultados_1_cqso).to_csv('resultados/sam_resultados_1_cqso.csv')
pd.DataFrame(sam_resultados_mse_treino_1_cqso).to_csv('resultados/sam_resultados_mse_treino_1_cqso.csv')
pd.DataFrame(sam_resultados_mse_teste_1_cqso).to_csv('resultados/sam_resultados_mse_teste_1_cqso.csv')


# ### Cenário II - Sunspot
# 
# Testando Cenário II
# 
# * w = 60
# * s = 20
# * f = 100


w = 60 # tamanho da janela
s = 20 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(y)-w)+f)
print('Quantidade total de iteracoes: ', T)

quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
# #backprop
print('BACKPROP')
sam_mse_treino_2_backprop, sam_mse_val_2_backprop, sam_mse_teste_2_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                  perc_treino=0.54, perc_val=0.24)
sam_resultados_2_backprop, sam_resultados_mse_treino_2_backprop, sam_resultados_mse_teste_2_backprop = avaliacao_resultados(sam_mse_treino_2_backprop, 
                                                                                                                             sam_mse_val_2_backprop, 
                                                                                                                             sam_mse_teste_2_backprop, 
                                                                                                                             f, quantidade_janelas, 
                                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(sam_resultados_2_backprop).to_csv('resultados/sam_resultados_2_backprop.csv')
pd.DataFrame(sam_resultados_mse_treino_2_backprop).to_csv('resultados/sam_resultados_mse_treino_2_backprop.csv')
pd.DataFrame(sam_resultados_mse_teste_2_backprop).to_csv('resultados/sam_resultados_mse_teste_2_backprop.csv')



print('PSO')
# # pso
tic = time.time()
sam_mse_treino_2_pso, sam_mse_val_2_pso, sam_mse_teste_2_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                   perc_treino=0.54, perc_val=0.24)
sam_resultados_2_pso, sam_resultados_mse_treino_2_pso, sam_resultados_mse_teste_2_pso = avaliacao_resultados(sam_mse_treino_2_pso, 
                                                                                                  sam_mse_val_2_pso,sam_mse_teste_2_pso, 
                                                                                                              f, quantidade_janelas, 
                                                                                                              execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(sam_resultados_2_pso).to_csv('resultados/sam_resultados_2_pso.csv')
pd.DataFrame(sam_resultados_mse_treino_2_pso).to_csv('resultados/sam_resultados_mse_treino_2_pso.csv')
pd.DataFrame(sam_resultados_mse_teste_2_pso).to_csv('resultados/sam_resultados_mse_teste_2_pso.csv')

print('CQSO')
# cqso
tic = time.time()
sam_mse_treino_2_cqso, sam_mse_val_2_cqso, sam_mse_teste_2_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sam_resultados_2_cqso, sam_resultados_mse_treino_2_cqso, sam_resultados_mse_teste_2_cqso = avaliacao_resultados(sam_mse_treino_2_cqso, 
                                                                                                             sam_mse_val_2_cqso,
                                                                                                             sam_mse_teste_2_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(sam_resultados_2_cqso).to_csv('resultados/sam_resultados_2_cqso.csv')
pd.DataFrame(sam_resultados_mse_treino_2_cqso).to_csv('resultados/sam_resultados_mse_treino_2_cqso.csv')
pd.DataFrame(sam_resultados_mse_teste_2_cqso).to_csv('resultados/sam_resultados_mse_teste_2_cqso.csv')


# ### Cenário III - Sunspot
# 
# Testando Cenário III
# 
# * w = 60
# * s = 40
# * f = 150

w = 60 # tamanho da janela
s = 40 # tamanho do passo
f = 150 # quantidade de iteracões para a janela
T = int(f/s*(len(sunspot)-w)+f)
print('Quantidade total de iteracões: ', T)
quantidade_janelas = int((len(y) - w)/s)

tic = time.time()
# #backprop
print('BACKPROP')
sam_mse_treino_3_backprop, sam_mse_val_3_backprop, sam_mse_teste_3_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                  perc_treino=0.54, perc_val=0.24)
sam_resultados_3_backprop, sam_resultados_mse_treino_3_backprop, sam_resultados_mse_teste_3_backprop = avaliacao_resultados(sam_mse_treino_3_backprop, 
                                                                                                                             sam_mse_val_3_backprop, 
                                                                                                                             sam_mse_teste_3_backprop, 
                                                                                                                             f, quantidade_janelas, 
                                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(sam_resultados_3_backprop).to_csv('resultados/sam_resultados_3_backprop.csv')
pd.DataFrame(sam_resultados_mse_treino_3_backprop).to_csv('resultados/sam_resultados_mse_treino_3_backprop.csv')
pd.DataFrame(sam_resultados_mse_teste_3_backprop).to_csv('resultados/sam_resultados_mse_teste_3_backprop.csv')

print('PSO')
# # pso
tic = time.time()
sam_mse_treino_3_pso, sam_mse_val_3_pso, sam_mse_teste_3_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                   perc_treino=0.54, perc_val=0.24)
sam_resultados_3_pso, sam_resultados_mse_treino_3_pso, sam_resultados_mse_teste_3_pso = avaliacao_resultados(sam_mse_treino_3_pso, 
                                                                                                  sam_mse_val_3_pso,sam_mse_teste_3_pso, 
                                                                                                              f, quantidade_janelas, 
                                                                                                              execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(sam_resultados_3_pso).to_csv('resultados/sam_resultados_3_pso.csv')
pd.DataFrame(sam_resultados_mse_treino_3_pso).to_csv('resultados/sam_resultados_mse_treino_3_pso.csv')
pd.DataFrame(sam_resultados_mse_teste_3_pso).to_csv('resultados/sam_resultados_mse_teste_3_pso.csv')

print('CQSO')
# cqso
tic = time.time()
sam_mse_treino_3_cqso, sam_mse_val_3_cqso, sam_mse_teste_3_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sam_resultados_3_cqso, sam_resultados_mse_treino_3_cqso, sam_resultados_mse_teste_3_cqso = avaliacao_resultados(sam_mse_treino_3_cqso, 
                                                                                                             sam_mse_val_3_cqso,
                                                                                                             sam_mse_teste_3_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(sam_resultados_3_cqso).to_csv('resultados/sam_resultados_3_cqso.csv')
pd.DataFrame(sam_resultados_mse_treino_3_cqso).to_csv('resultados/sam_resultados_mse_treino_3_cqso.csv')
pd.DataFrame(sam_resultados_mse_teste_3_cqso).to_csv('resultados/sam_resultados_mse_teste_3_cqso.csv')


# ### Cenário IV - Sunspot
# 
# Testando Cenário IV
# 
# * w = 60
# * s = 60
# * f = 100


w = 60 # tamanho da janela
s = 60 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(sunspot)-w)+f)
print('Quantidade total de iteracões: ', T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
# #backprop
print('BACKPROP')
sam_mse_treino_4_backprop, sam_mse_val_4_backprop, sam_mse_teste_4_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                  perc_treino=0.54, perc_val=0.24)
sam_resultados_4_backprop, sam_resultados_mse_treino_4_backprop, sam_resultados_mse_teste_4_backprop = avaliacao_resultados(sam_mse_treino_4_backprop, 
                                                                                                                             sam_mse_val_4_backprop, 
                                                                                                                             sam_mse_teste_4_backprop, 
                                                                                                                             f, quantidade_janelas, 
                                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(sam_resultados_4_backprop).to_csv('resultados/sam_resultados_4_backprop.csv')
pd.DataFrame(sam_resultados_mse_treino_4_backprop).to_csv('resultados/sam_resultados_mse_treino_4_backprop.csv')
pd.DataFrame(sam_resultados_mse_teste_4_backprop).to_csv('resultados/sam_resultados_mse_teste_4_backprop.csv')


print('PSO')
# # pso
tic = time.time()
sam_mse_treino_4_pso, sam_mse_val_4_pso, sam_mse_teste_4_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                   perc_treino=0.54, perc_val=0.24)
sam_resultados_4_pso, sam_resultados_mse_treino_4_pso, sam_resultados_mse_teste_4_pso = avaliacao_resultados(sam_mse_treino_4_pso, 
                                                                                                  sam_mse_val_4_pso,sam_mse_teste_4_pso, 
                                                                                                              f, quantidade_janelas, 
                                                                                                              execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(sam_resultados_4_pso).to_csv('resultados/sam_resultados_4_pso.csv')
pd.DataFrame(sam_resultados_mse_treino_4_pso).to_csv('resultados/sam_resultados_mse_treino_4_pso.csv')
pd.DataFrame(sam_resultados_mse_teste_4_pso).to_csv('resultados/sam_resultados_mse_teste_4_pso.csv')

print('CQSO')
# cqso
tic = time.time()
sam_mse_treino_4_cqso, sam_mse_val_4_cqso, sam_mse_teste_4_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sam_resultados_4_cqso, sam_resultados_mse_treino_4_cqso, sam_resultados_mse_teste_4_pso = avaliacao_resultados(sam_mse_treino_4_cqso, 
                                                                                                             sam_mse_val_4_cqso,
                                                                                                             sam_mse_teste_4_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(sam_resultados_4_cqso).to_csv('resultados/sam_resultados_4_cqso.csv')
pd.DataFrame(sam_resultados_mse_treino_4_cqso).to_csv('resultados/sam_resultados_mse_treino_4_cqso.csv')
pd.DataFrame(sam_resultados_mse_teste_4_cqso).to_csv('resultados/sam_resultados_mse_teste_4_cqso.csv')


# ## 2. International Airline Time Series (IAP)
# 
# * 144 obs
# * jan 1949 to dez 1960
# 
# Série mensal inputs: 12
# 


## Airline Passenger
airline = pd.read_csv('dados/airline_passengers.csv')
airline = airline['valor']
airline.head()


qtd_inputs = 12
airline_norm = normalizar_serie(airline)
X, y = split_sequence(airline_norm.values, qtd_inputs, 1)


# ### Cenário I - Airline
# 
# * w = 32
# * s = 5
# * f = 50

w = 32 # tamanho da janela
s = 5 # tamanho do passo
f = 50 # quantidade de iteracões para a janela
T = int(f/s*(len(airline)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)

tic = time.time()
#backprop
print('BACKPROP')
iap_mse_treino_1_backprop, iap_mse_val_1_backprop, iap_mse_teste_1_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
iap_resultados_1_backprop, iap_resultados_mse_treino_1_backprop, iap_resultados_mse_teste_1_backprop = avaliacao_resultados(iap_mse_treino_1_backprop, 
                                                                                                                            iap_mse_val_1_backprop, 
                                                                                                                            iap_mse_teste_1_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(iap_resultados_1_backprop).to_csv('resultados/iap_resultados_1_backprop.csv')
pd.DataFrame(iap_resultados_mse_treino_1_backprop).to_csv('resultados/iap_resultados_mse_treino_1_backprop.csv')
pd.DataFrame(iap_resultados_mse_teste_1_backprop).to_csv('resultados/iap_resultados_mse_teste_1_backprop.csv')



print('PSO')
# pso
tic = time.time()
iap_mse_treino_1_pso, iap_mse_val_1_pso, iap_mse_teste_1_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
iap_resultados_1_pso, iap_resultados_mse_treino_1_pso, iap_resultados_mse_teste_1_pso = avaliacao_resultados(iap_mse_treino_1_pso, 
                                                                                                             iap_mse_val_1_pso,
                                                                                                             iap_mse_teste_1_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(iap_resultados_1_pso).to_csv('resultados/iap_resultados_1_pso.csv')
pd.DataFrame(iap_resultados_mse_treino_1_pso).to_csv('resultados/iap_resultados_mse_treino_1_pso.csv')
pd.DataFrame(iap_resultados_mse_teste_1_pso).to_csv('resultados/iap_resultados_mse_teste_1_pso.csv')

print('CQSO')
# cqso
tic = time.time()
iap_mse_treino_1_cqso, iap_mse_val_1_cqso, iap_mse_teste_1_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
iap_resultados_1_cqso, iap_resultados_mse_treino_1_cqso, iap_resultados_mse_teste_1_cqso = avaliacao_resultados(iap_mse_treino_1_cqso, 
                                                                                                             iap_mse_val_1_cqso,
                                                                                                             iap_mse_teste_1_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(iap_resultados_1_cqso).to_csv('resultados/iap_resultados_1_cqso.csv')
pd.DataFrame(iap_resultados_mse_treino_1_cqso).to_csv('resultados/iap_resultados_mse_treino_1_cqso.csv')
pd.DataFrame(iap_resultados_mse_teste_1_cqso).to_csv('resultados/iap_resultados_mse_teste_1_cqso.csv')


# ### Cenário II - Airline
# 
# * w = 32
# * s = 10
# * f = 100

w = 32 # tamanho da janela
s = 10 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(airline)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
iap_mse_treino_2_backprop, iap_mse_val_2_backprop, iap_mse_teste_2_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
iap_resultados_2_backprop, iap_resultados_mse_treino_2_backprop, iap_resultados_mse_teste_2_backprop = avaliacao_resultados(iap_mse_treino_2_backprop, 
                                                                                                                            iap_mse_val_2_backprop, 
                                                                                                                            iap_mse_teste_2_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(iap_resultados_2_backprop).to_csv('resultados/iap_resultados_2_backprop.csv')
pd.DataFrame(iap_resultados_mse_treino_2_backprop).to_csv('resultados/iap_resultados_mse_treino_2_backprop.csv')
pd.DataFrame(iap_resultados_mse_teste_2_backprop).to_csv('resultados/iap_resultados_mse_teste_2_backprop.csv')


print('PSO')
# pso
tic = time.time()
iap_mse_treino_2_pso, iap_mse_val_2_pso, iap_mse_teste_2_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
iap_resultados_2_pso, iap_resultados_mse_treino_2_pso, iap_resultados_mse_teste_2_pso = avaliacao_resultados(iap_mse_treino_2_pso, 
                                                                                                 iap_mse_val_2_pso,iap_mse_teste_2_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(iap_resultados_2_pso).to_csv('resultados/iap_resultados_2_pso.csv')
pd.DataFrame(iap_resultados_mse_treino_2_pso).to_csv('resultados/iap_resultados_mse_treino_2_pso.csv')
pd.DataFrame(iap_resultados_mse_teste_2_pso).to_csv('resultados/iap_resultados_mse_teste_2_pso.csv')

print('CQSO')
# cqso
tic = time.time()
iap_mse_treino_2_cqso, iap_mse_val_2_cqso, iap_mse_teste_2_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
iap_resultados_2_cqso, iap_resultados_mse_treino_2_cqso, iap_resultados_mse_teste_2_cqso = avaliacao_resultados(iap_mse_treino_2_cqso, 
                                                                                                             iap_mse_val_2_cqso,
                                                                                                             iap_mse_teste_2_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(iap_resultados_2_cqso).to_csv('resultados/iap_resultados_2_cqso.csv')
pd.DataFrame(iap_resultados_mse_treino_2_cqso).to_csv('resultados/iap_resultados_mse_treino_2_cqso.csv')
pd.DataFrame(iap_resultados_mse_teste_2_cqso).to_csv('resultados/iap_resultados_mse_teste_2_cqso.csv')


# ### Cenário III - Airline
# 
# * w = 32
# * s = 25
# * f = 150

w = 32 # tamanho da janela
s = 25 # tamanho do passo
f = 150 # quantidade de iteracões para a janela
T = int(f/s*(len(airline)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)

tic = time.time()
#backprop
print('BACKPROP')
iap_mse_treino_3_backprop, iap_mse_val_3_backprop, iap_mse_teste_3_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
iap_resultados_3_backprop, iap_resultados_mse_treino_3_backprop, iap_resultados_mse_teste_3_backprop = avaliacao_resultados(iap_mse_treino_3_backprop, 
                                                                                                                            iap_mse_val_3_backprop, 
                                                                                                                            iap_mse_teste_3_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(iap_resultados_3_backprop).to_csv('resultados/iap_resultados_3_backprop.csv')
pd.DataFrame(iap_resultados_mse_treino_3_backprop).to_csv('resultados/iap_resultados_mse_treino_3_backprop.csv')
pd.DataFrame(iap_resultados_mse_teste_3_backprop).to_csv('resultados/iap_resultados_mse_teste_3_backprop.csv')


print('PSO')
# pso
tic = time.time()
iap_mse_treino_3_pso, iap_mse_val_3_pso, iap_mse_teste_3_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
iap_resultados_3_pso, iap_resultados_mse_treino_3_pso, iap_resultados_mse_teste_3_pso = avaliacao_resultados(iap_mse_treino_3_pso, 
                                                                                                             iap_mse_val_3_pso,
                                                                                                             iap_mse_teste_3_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(iap_resultados_3_pso).to_csv('resultados/iap_resultados_3_pso.csv')
pd.DataFrame(iap_resultados_mse_treino_3_pso).to_csv('resultados/iap_resultados_mse_treino_3_pso.csv')
pd.DataFrame(iap_resultados_mse_teste_3_pso).to_csv('resultados/iap_resultados_mse_teste_3_pso.csv')

print('CQSO')
# cqso
tic = time.time()
iap_mse_treino_3_cqso, iap_mse_val_3_cqso, iap_mse_teste_3_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
iap_resultados_3_cqso, iap_resultados_mse_treino_3_cqso, iap_resultados_mse_teste_3_cqso = avaliacao_resultados(iap_mse_treino_3_cqso, 
                                                                                                             iap_mse_val_3_cqso,
                                                                                                             iap_mse_teste_3_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(iap_resultados_3_cqso).to_csv('resultados/iap_resultados_3_cqso.csv')
pd.DataFrame(iap_resultados_mse_treino_3_cqso).to_csv('resultados/iap_resultados_mse_treino_3_cqso.csv')
pd.DataFrame(iap_resultados_mse_teste_3_cqso).to_csv('resultados/iap_resultados_mse_teste_3_cqso.csv')


# ### Cenário IV - Airline
# 
# * w = 32
# * s = 32
# * f = 100

w = 32
s = 32
f = 100
T = int(f/s*(len(airline)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
iap_mse_treino_4_backprop, iap_mse_val_4_backprop, iap_mse_teste_4_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
iap_resultados_4_backprop, iap_resultados_mse_treino_4_backprop, iap_resultados_mse_teste_4_backprop = avaliacao_resultados(iap_mse_treino_4_backprop, 
                                                                                                                            iap_mse_val_4_backprop, 
                                                                                                                            iap_mse_teste_4_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(iap_resultados_4_backprop).to_csv('resultados/iap_resultados_4_backprop.csv')
pd.DataFrame(iap_resultados_mse_treino_4_backprop).to_csv('resultados/iap_resultados_mse_treino_4_backprop.csv')
pd.DataFrame(iap_resultados_mse_teste_4_backprop).to_csv('resultados/iap_resultados_mse_teste_4_backprop.csv')


print('PSO')
# pso
tic = time.time()
iap_mse_treino_4_pso, iap_mse_val_4_pso, iap_mse_teste_4_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
iap_resultados_4_pso, iap_resultados_mse_treino_4_pso, iap_resultados_mse_teste_4_pso = avaliacao_resultados(iap_mse_treino_4_pso, 
                                                                                                             iap_mse_val_4_pso,
                                                                                                             iap_mse_teste_4_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(iap_resultados_4_pso).to_csv('resultados/iap_resultados_4_pso.csv')
pd.DataFrame(iap_resultados_mse_treino_4_pso).to_csv('resultados/iap_resultados_mse_treino_4_pso.csv')
pd.DataFrame(iap_resultados_mse_teste_4_pso).to_csv('resultados/iap_resultados_mse_teste_4_pso.csv')

print('CQSO')
# cqso
tic = time.time()
iap_mse_treino_4_cqso, iap_mse_val_4_cqso, iap_mse_teste_4_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
iap_resultados_4_cqso, iap_resultados_mse_treino_4_cqso, iap_resultados_mse_teste_4_cqso = avaliacao_resultados(iap_mse_treino_4_cqso, 
                                                                                                             iap_mse_val_4_cqso,
                                                                                                             iap_mse_teste_4_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(iap_resultados_4_cqso).to_csv('resultados/iap_resultados_4_cqso.csv')
pd.DataFrame(iap_resultados_mse_treino_4_cqso).to_csv('resultados/iap_resultados_mse_treino_4_cqso.csv')
pd.DataFrame(iap_resultados_mse_teste_4_cqso).to_csv('resultados/iap_resultados_mse_teste_4_cqso.csv')


# ## 3. Australian wine sales time series (AWS)
# 
# * 187 obs. 
# * jan 1980 to july 1995
# 
# Série mensal inputs: 12


## AWS
aws = pd.read_csv('dados/wine_sales.csv')
aws = aws['valor']
aws.head()


qtd_inputs = 12
aws_norm = normalizar_serie(aws)
X, y = split_sequence(aws_norm.values, qtd_inputs, 1)


# ### Cenário I - AWS
# 
# * w = 42 
# * s = 5
# * f = 50


w = 42 # tamanho da janela
s = 5 # tamanho do passo
f = 50 # quantidade de iteracões para a janela
T = int(f/s*(len(aws)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
aws_mse_treino_1_backprop, aws_mse_val_1_backprop, aws_mse_teste_1_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
aws_resultados_1_backprop, aws_resultados_mse_treino_1_backprop, aws_resultados_mse_teste_1_backprop = avaliacao_resultados(aws_mse_treino_1_backprop, 
                                                                                                                            aws_mse_val_1_backprop, 
                                                                                                                            aws_mse_teste_1_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(aws_resultados_1_backprop).to_csv('resultados/aws_resultados_1_backprop.csv')
pd.DataFrame(aws_resultados_mse_treino_1_backprop).to_csv('resultados/aws_resultados_mse_treino_1_backprop.csv')
pd.DataFrame(aws_resultados_mse_teste_1_backprop).to_csv('resultados/aws_resultados_mse_teste_1_backprop.csv')

print('PSO')
# pso
tic = time.time()
aws_mse_treino_1_pso, aws_mse_val_1_pso, aws_mse_teste_1_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
aws_resultados_1_pso, aws_resultados_mse_treino_1_pso, aws_resultados_mse_teste_1_pso = avaliacao_resultados(aws_mse_treino_1_pso, 
                                                                                                             aws_mse_val_1_pso,
                                                                                                             aws_mse_teste_1_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(aws_resultados_1_pso).to_csv('resultados/aws_resultados_1_pso.csv')
pd.DataFrame(aws_resultados_mse_treino_1_pso).to_csv('resultados/aws_resultados_mse_treino_1_pso.csv')
pd.DataFrame(aws_resultados_mse_teste_1_pso).to_csv('resultados/aws_resultados_mse_teste_1_pso.csv')

print('CQSO')
# cqso
tic = time.time()
aws_mse_treino_1_cqso, aws_mse_val_1_cqso, aws_mse_teste_1_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
aws_resultados_1_cqso, aws_resultados_mse_treino_1_cqso, aws_resultados_mse_teste_1_cqso = avaliacao_resultados(aws_mse_treino_1_cqso, 
                                                                                                             aws_mse_val_1_cqso,
                                                                                                             aws_mse_teste_1_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(aws_resultados_1_cqso).to_csv('resultados/aws_resultados_1_cqso.csv')
pd.DataFrame(aws_resultados_mse_treino_1_cqso).to_csv('resultados/aws_resultados_mse_treino_1_cqso.csv')
pd.DataFrame(aws_resultados_mse_teste_1_cqso).to_csv('resultados/aws_resultados_mse_teste_1_cqso.csv')


# ### Cenário II - AWS
# 
# * w = 42
# * s = 20
# * f = 100

w = 42 # tamanho da janela
s = 20 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(aws)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)

tic = time.time()
#backprop
print('BACKPROP')
aws_mse_treino_2_backprop, aws_mse_val_2_backprop, aws_mse_teste_2_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
aws_resultados_2_backprop, aws_resultados_mse_treino_2_backprop, aws_resultados_mse_teste_2_backprop = avaliacao_resultados(aws_mse_treino_2_backprop, 
                                                                                                                            aws_mse_val_2_backprop, 
                                                                                                                            aws_mse_teste_2_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(aws_resultados_2_backprop).to_csv('resultados/aws_resultados_2_backprop.csv')
pd.DataFrame(aws_resultados_mse_treino_2_backprop).to_csv('resultados/aws_resultados_mse_treino_2_backprop.csv')
pd.DataFrame(aws_resultados_mse_teste_2_backprop).to_csv('resultados/aws_resultados_mse_teste_2_backprop.csv')

print('PSO')
# pso
tic = time.time()
aws_mse_treino_2_pso, aws_mse_val_2_pso, aws_mse_teste_2_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
aws_resultados_2_pso, aws_resultados_mse_treino_2_pso, aws_resultados_mse_teste_2_pso = avaliacao_resultados(aws_mse_treino_2_pso, 
                                                                                                             aws_mse_val_2_pso,
                                                                                                             aws_mse_teste_2_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(aws_resultados_2_pso).to_csv('resultados/aws_resultados_2_pso.csv')
pd.DataFrame(aws_resultados_mse_treino_2_pso).to_csv('resultados/aws_resultados_mse_treino_2_pso.csv')
pd.DataFrame(aws_resultados_mse_teste_2_pso).to_csv('resultados/aws_resultados_mse_teste_2_pso.csv')

print('CQSO')
# cqso
tic = time.time()
aws_mse_treino_2_cqso, aws_mse_val_2_cqso, aws_mse_teste_2_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
aws_resultados_2_cqso, aws_resultados_mse_treino_2_cqso, aws_resultados_mse_teste_2_cqso = avaliacao_resultados(aws_mse_treino_2_cqso, 
                                                                                                             aws_mse_val_2_cqso,
                                                                                                             aws_mse_teste_2_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(aws_resultados_2_cqso).to_csv('resultados/aws_resultados_2_cqso.csv')
pd.DataFrame(aws_resultados_mse_treino_2_cqso).to_csv('resultados/aws_resultados_mse_treino_2_cqso.csv')
pd.DataFrame(aws_resultados_mse_teste_2_cqso).to_csv('resultados/aws_resultados_mse_teste_2_cqso.csv')


# ### Cenário III - AWS
# 
# * w = 42
# * s = 35
# * f = 150

w = 42 # tamanho da janela
s = 35 # tamanho do passo
f = 150 # quantidade de iteracões para a janela
T = int(f/s*(len(aws)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
aws_mse_treino_3_backprop, aws_mse_val_3_backprop, aws_mse_teste_3_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
aws_resultados_3_backprop, aws_resultados_mse_treino_3_backprop, aws_resultados_mse_teste_3_backprop = avaliacao_resultados(aws_mse_treino_3_backprop, 
                                                                                                                            aws_mse_val_3_backprop, 
                                                                                                                            aws_mse_teste_3_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 

tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(aws_resultados_3_backprop).to_csv('resultados/aws_resultados_3_backprop.csv')
pd.DataFrame(aws_resultados_mse_treino_3_backprop).to_csv('resultados/aws_resultados_mse_treino_3_backprop.csv')
pd.DataFrame(aws_resultados_mse_teste_3_backprop).to_csv('resultados/aws_resultados_mse_teste_3_backprop.csv')


print('PSO')
# pso
tic = time.time()
aws_mse_treino_3_pso, aws_mse_val_3_pso, aws_mse_teste_3_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
aws_resultados_3_pso, aws_resultados_mse_treino_3_pso, aws_resultados_mse_teste_3_pso = avaliacao_resultados(aws_mse_treino_3_pso, 
                                                                                                             aws_mse_val_3_pso,
                                                                                                             aws_mse_teste_3_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(aws_resultados_3_pso).to_csv('resultados/aws_resultados_3_pso.csv')
pd.DataFrame(aws_resultados_mse_treino_3_pso).to_csv('resultados/aws_resultados_mse_treino_3_pso.csv')
pd.DataFrame(aws_resultados_mse_teste_3_pso).to_csv('resultados/aws_resultados_mse_teste_3_pso.csv')

print('CQSO')
# cqso
tic = time.time()
aws_mse_treino_3_cqso, aws_mse_val_3_cqso, aws_mse_teste_3_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
aws_resultados_3_cqso, aws_resultados_mse_treino_3_cqso, aws_resultados_mse_teste_3_cqso = avaliacao_resultados(aws_mse_treino_3_cqso, 
                                                                                                             aws_mse_val_3_cqso,
                                                                                                             aws_mse_teste_3_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(aws_resultados_3_cqso).to_csv('resultados/aws_resultados_3_cqso.csv')
pd.DataFrame(aws_resultados_mse_treino_3_cqso).to_csv('resultados/aws_resultados_mse_treino_3_cqso.csv')
pd.DataFrame(aws_resultados_mse_teste_3_cqso).to_csv('resultados/aws_resultados_mse_teste_3_cqso.csv')


# ### Cenário IV - AWS
# 
# * w = 42
# * s = 32
# * f = 100

w = 42 # tamanho da janela
s = 32 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(aws)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
aws_mse_treino_4_backprop, aws_mse_val_4_backprop, aws_mse_teste_4_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
aws_resultados_4_backprop, aws_resultados_mse_treino_4_backprop, aws_resultados_mse_teste_4_backprop = avaliacao_resultados(aws_mse_treino_4_backprop, 
                                                                                                                            aws_mse_val_4_backprop, 
                                                                                                                            aws_mse_teste_4_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(aws_resultados_4_backprop).to_csv('resultados/aws_resultados_4_backprop.csv')
pd.DataFrame(aws_resultados_mse_treino_4_backprop).to_csv('resultados/aws_resultados_mse_treino_4_backprop.csv')
pd.DataFrame(aws_resultados_mse_teste_4_backprop).to_csv('resultados/aws_resultados_mse_teste_4_backprop.csv')



print('PSO')
# pso
tic = time.time()
aws_mse_treino_4_pso, aws_mse_val_4_pso, aws_mse_teste_4_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
aws_resultados_4_pso, aws_resultados_mse_treino_4_pso, aws_resultados_mse_teste_4_pso = avaliacao_resultados(aws_mse_treino_4_pso, 
                                                                                                             aws_mse_val_4_pso,
                                                                                                             aws_mse_teste_4_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(aws_resultados_4_pso).to_csv('resultados/aws_resultados_4_pso.csv')
pd.DataFrame(aws_resultados_mse_treino_4_pso).to_csv('resultados/aws_resultados_mse_treino_4_pso.csv')
pd.DataFrame(aws_resultados_mse_teste_4_pso).to_csv('resultados/aws_resultados_mse_teste_4_pso.csv')

print('CQSO')
# cqso
tic = time.time()
aws_mse_treino_4_cqso, aws_mse_val_4_cqso, aws_mse_teste_4_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
aws_resultados_4_cqso, aws_resultados_mse_treino_4_cqso, aws_resultados_mse_teste_4_cqso = avaliacao_resultados(aws_mse_treino_4_cqso, 
                                                                                                             aws_mse_val_4_cqso,
                                                                                                             aws_mse_teste_4_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(aws_resultados_4_cqso).to_csv('resultados/aws_resultados_4_cqso.csv')
pd.DataFrame(aws_resultados_mse_treino_4_cqso).to_csv('resultados/aws_resultados_mse_treino_4_cqso.csv')
pd.DataFrame(aws_resultados_mse_teste_4_cqso).to_csv('resultados/aws_resultados_mse_teste_4_cqso.csv')


# ## 4. Standard and poor 500 indexes (S&P)
# 
# * 388 obs. 
# * Quarterly
# *  1990 to july 1996
# 
# Série trimestral inputs: 4

## S&P
sp500 = pd.read_csv('dados/sp500.csv')
sp500 = sp500['valor']
sp500.head()


qtd_inputs = 4
aws_norm = normalizar_serie(aws)
X, y = split_sequence(aws_norm.values, qtd_inputs, 1)


# ### Cenário I - S&P
# 
# * w = 58
# * s = 10
# * f = 50


w = 58 # tamanho da janela
s = 10 # tamanho do passo
f = 50 # quantidade de iteracões para a janela
T = int(f/s*(len(sp500)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)

tic = time.time()
#backprop
print('BACKPROP')
sp500_mse_treino_1_backprop, sp500_mse_val_1_backprop, sp500_mse_teste_1_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
sp500_resultados_1_backprop, sp500_resultados_mse_treino_1_backprop, sp500_resultados_mse_teste_1_backprop = avaliacao_resultados(sp500_mse_treino_1_backprop, 
                                                                                                                            sp500_mse_val_1_backprop, 
                                                                                                                            sp500_mse_teste_1_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(sp500_resultados_1_backprop).to_csv('resultados/sp500_resultados_1_backprop.csv')
pd.DataFrame(sp500_resultados_mse_treino_1_backprop).to_csv('resultados/sp500_resultados_mse_treino_1_backprop.csv')
pd.DataFrame(sp500_resultados_mse_teste_1_backprop).to_csv('resultados/sp500_resultados_mse_teste_1_backprop.csv')


print('PSO')
# pso
tic = time.time()
sp500_mse_treino_1_pso, sp500_mse_val_1_pso, sp500_mse_teste_1_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sp500_resultados_1_pso, sp500_resultados_mse_treino_1_pso, sp500_resultados_mse_teste_1_pso = avaliacao_resultados(sp500_mse_treino_1_pso, 
                                                                                                             sp500_mse_val_1_pso,
                                                                                                             sp500_mse_teste_1_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(sp500_resultados_1_pso).to_csv('resultados/sp500_resultados_1_pso.csv')
pd.DataFrame(sp500_resultados_mse_treino_1_pso).to_csv('resultados/sp500_resultados_mse_treino_1_pso.csv')
pd.DataFrame(sp500_resultados_mse_teste_1_pso).to_csv('resultados/sp500_resultados_mse_teste_1_pso.csv')

print('CQSO')
# cqso
tic = time.time()
sp500_mse_treino_1_cqso, sp500_mse_val_1_cqso, sp500_mse_teste_1_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sp500_resultados_1_cqso, sp500_resultados_mse_treino_1_cqso, sp500_resultados_mse_teste_1_cqso = avaliacao_resultados(sp500_mse_treino_1_cqso, 
                                                                                                             sp500_mse_val_1_cqso,
                                                                                                             sp500_mse_teste_1_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(sp500_resultados_1_cqso).to_csv('resultados/sp500_resultados_1_cqso.csv')
pd.DataFrame(sp500_resultados_mse_treino_1_cqso).to_csv('resultados/sp500_resultados_mse_treino_1_cqso.csv')
pd.DataFrame(sp500_resultados_mse_teste_1_cqso).to_csv('resultados/sp500_resultados_mse_teste_1_cqso.csv')


# ### Cenário II - S&P
# 
# * w = 58
# * s = 20
# * f = 100


w = 58 # tamanho da janela
s = 20 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(aws)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
sp500_mse_treino_2_backprop, sp500_mse_val_2_backprop, sp500_mse_teste_2_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
sp500_resultados_2_backprop, sp500_resultados_mse_treino_2_backprop, sp500_resultados_mse_teste_2_backprop = avaliacao_resultados(sp500_mse_treino_2_backprop, 
                                                                                                                            sp500_mse_val_2_backprop, 
                                                                                                                            sp500_mse_teste_2_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(sp500_resultados_2_backprop).to_csv('resultados/sp500_resultados_2_backprop.csv')
pd.DataFrame(sp500_resultados_mse_treino_2_backprop).to_csv('resultados/sp500_resultados_mse_treino_2_backprop.csv')
pd.DataFrame(sp500_resultados_mse_teste_2_backprop).to_csv('resultados/sp500_resultados_mse_teste_2_backprop.csv')


print('PSO')
# pso
tic = time.time()
sp500_mse_treino_2_pso, sp500_mse_val_2_pso, sp500_mse_teste_2_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sp500_resultados_2_pso, sp500_resultados_mse_treino_2_pso, sp500_resultados_mse_teste_2_pso = avaliacao_resultados(sp500_mse_treino_2_pso, 
                                                                                                             sp500_mse_val_2_pso,
                                                                                                             sp500_mse_teste_2_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(sp500_resultados_2_pso).to_csv('resultados/sp500_resultados_2_pso.csv')
pd.DataFrame(sp500_resultados_mse_treino_2_pso).to_csv('resultados/sp500_resultados_mse_treino_2_pso.csv')
pd.DataFrame(sp500_resultados_mse_teste_2_pso).to_csv('resultados/sp500_resultados_mse_teste_2_pso.csv')

print('CQSO')
# cqso
tic = time.time()
sp500_mse_treino_2_cqso, sp500_mse_val_2_cqso, sp500_mse_teste_2_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sp500_resultados_2_cqso, sp500_resultados_mse_treino_2_cqso, sp500_resultados_mse_teste_2_cqso = avaliacao_resultados(sp500_mse_treino_2_cqso, 
                                                                                                             sp500_mse_val_2_cqso,
                                                                                                             sp500_mse_teste_2_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(sp500_resultados_2_cqso).to_csv('resultados/sp500_resultados_2_cqso.csv')
pd.DataFrame(sp500_resultados_mse_treino_2_cqso).to_csv('resultados/sp500_resultados_mse_treino_2_cqso.csv')
pd.DataFrame(sp500_resultados_mse_teste_2_cqso).to_csv('resultados/sp500_resultados_mse_teste_2_cqso.csv')


# ### Cenário III - S&P
# 
# * w = 58
# * s = 40
# * f = 150

w = 58 # tamanho da janela
s = 40 # tamanho do passo
f = 150 # quantidade de iteracões para a janela
T = int(f/s*(len(aws)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
sp500_mse_treino_3_backprop, sp500_mse_val_3_backprop, sp500_mse_teste_3_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
sp500_resultados_3_backprop, sp500_resultados_mse_treino_3_backprop, sp500_resultados_mse_teste_3_backprop = avaliacao_resultados(sp500_mse_treino_3_backprop, 
                                                                                                                            sp500_mse_val_3_backprop, 
                                                                                                                            sp500_mse_teste_3_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(sp500_resultados_3_backprop).to_csv('resultados/sp500_resultados_3_backprop.csv')
pd.DataFrame(sp500_resultados_mse_treino_3_backprop).to_csv('resultados/sp500_resultados_mse_treino_3_backprop.csv')
pd.DataFrame(sp500_resultados_mse_teste_3_backprop).to_csv('resultados/sp500_resultados_mse_teste_3_backprop.csv')


print('PSO')
# pso
tic = time.time()
sp500_mse_treino_3_pso, sp500_mse_val_3_pso, sp500_mse_teste_3_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sp500_resultados_3_pso, sp500_resultados_mse_treino_3_pso, sp500_resultados_mse_teste_3_pso = avaliacao_resultados(sp500_mse_treino_3_pso, 
                                                                                                             sp500_mse_val_3_pso,
                                                                                                             sp500_mse_teste_3_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(sp500_resultados_3_pso).to_csv('resultados/sp500_resultados_3_pso.csv')
pd.DataFrame(sp500_resultados_mse_treino_3_pso).to_csv('resultados/sp500_resultados_mse_treino_3_pso.csv')
pd.DataFrame(sp500_resultados_mse_teste_3_pso).to_csv('resultados/sp500_resultados_mse_teste_3_pso.csv')


print('CQSO')
# cqso
tic = time.time()
sp500_mse_treino_3_cqso, sp500_mse_val_3_cqso, sp500_mse_teste_3_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sp500_resultados_3_cqso, sp500_resultados_mse_treino_3_cqso, sp500_resultados_mse_teste_3_cqso = avaliacao_resultados(sp500_mse_treino_3_cqso, 
                                                                                                             sp500_mse_val_3_cqso,
                                                                                                             sp500_mse_teste_3_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(sp500_resultados_3_cqso).to_csv('resultados/sp500_resultados_3_cqso.csv')
pd.DataFrame(sp500_resultados_mse_treino_3_cqso).to_csv('resultados/sp500_resultados_mse_treino_3_cqso.csv')
pd.DataFrame(sp500_resultados_mse_teste_3_cqso).to_csv('resultados/sp500_resultados_mse_teste_3_cqso.csv')


# ### Cenário IV - S&P
# 
# * w = 58
# * s = 58
# * f = 100

w = 58 # tamanho da janela
s = 58 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(aws)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)

tic = time.time()
#backprop
print('BACKPROP')
sp500_mse_treino_4_backprop, sp500_mse_val_4_backprop, sp500_mse_teste_4_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
sp500_resultados_4_backprop, sp500_resultados_mse_treino_4_backprop, sp500_resultados_mse_teste_4_backprop = avaliacao_resultados(sp500_mse_treino_4_backprop, 
                                                                                                                            sp500_mse_val_4_backprop, 
                                                                                                                            sp500_mse_teste_4_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(sp500_resultados_4_backprop).to_csv('resultados/sp500_resultados_4_backprop.csv')
pd.DataFrame(sp500_resultados_mse_treino_4_backprop).to_csv('resultados/sp500_resultados_mse_treino_4_backprop.csv')
pd.DataFrame(sp500_resultados_mse_teste_4_backprop).to_csv('resultados/sp500_resultados_mse_teste_4_backprop.csv')




print('PSO')
# pso
tic = time.time()
sp500_mse_treino_4_pso, sp500_mse_val_4_pso, sp500_mse_teste_4_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sp500_resultados_4_pso, sp500_resultados_mse_treino_4_pso, sp500_resultados_mse_teste_4_pso = avaliacao_resultados(sp500_mse_treino_4_pso, 
                                                                                                             sp500_mse_val_4_pso,
                                                                                                             sp500_mse_teste_4_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(sp500_resultados_4_pso).to_csv('resultados/sp500_resultados_4_pso.csv')
pd.DataFrame(sp500_resultados_mse_treino_4_pso).to_csv('resultados/sp500_resultados_mse_treino_4_pso.csv')
pd.DataFrame(sp500_resultados_mse_teste_4_pso).to_csv('resultados/sp500_resultados_mse_teste_4_pso.csv')

print('CQSO')
# cqso
tic = time.time()
sp500_mse_treino_4_cqso, sp500_mse_val_4_cqso, sp500_mse_teste_4_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
sp500_resultados_4_cqso, sp500_resultados_mse_treino_4_cqso, sp500_resultados_mse_teste_4_cqso = avaliacao_resultados(sp500_mse_treino_4_cqso, 
                                                                                                             sp500_mse_val_4_cqso,
                                                                                                             sp500_mse_teste_4_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(sp500_resultados_4_cqso).to_csv('resultados/sp500_resultados_4_cqso.csv')
pd.DataFrame(sp500_resultados_mse_treino_4_cqso).to_csv('resultados/sp500_resultados_mse_treino_4_cqso.csv')
pd.DataFrame(sp500_resultados_mse_teste_4_cqso).to_csv('resultados/sp500_resultados_mse_teste_4_cqso.csv')


# ## 5. US death time series (USD)
# 
# * 72 obs. 
# * Jan 1973 to Dez 1978
# 
# Série mensal inputs: 12

## USD
usd = pd.read_csv('dados/usa_accident_death.csv')
usd = usd['valor']
usd.head()


qtd_inputs = 12
usd_norm = normalizar_serie(usd)
X, y = split_sequence(usd_norm.values, qtd_inputs, 1)


# ### Cenário I - USD
# 
# * w = 20
# * s = 2
# * f = 50


w = 20 # tamanho da janela
s = 2 # tamanho do passo
f = 50 # quantidade de iteracões para a janela
T = int(f/s*(len(usd)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
usd_mse_treino_1_backprop, usd_mse_val_1_backprop, usd_mse_teste_1_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
usd_resultados_1_backprop, usd_resultados_mse_treino_1_backprop, usd_resultados_mse_teste_1_backprop = avaliacao_resultados(usd_mse_treino_1_backprop, 
                                                                                                                            usd_mse_val_1_backprop, 
                                                                                                                            usd_mse_teste_1_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(usd_resultados_1_backprop).to_csv('resultados/usd_resultados_1_backprop.csv')
pd.DataFrame(usd_resultados_mse_treino_1_backprop).to_csv('resultados/usd_resultados_mse_treino_1_backprop.csv')
pd.DataFrame(usd_resultados_mse_teste_1_backprop).to_csv('resultados/usd_resultados_mse_teste_1_backprop.csv')


print('PSO')
# pso
tic = time.time()
usd_mse_treino_1_pso, usd_mse_val_1_pso, usd_mse_teste_1_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
usd_resultados_1_pso, usd_resultados_mse_treino_1_pso, usd_resultados_mse_teste_1_pso = avaliacao_resultados(usd_mse_treino_1_pso, 
                                                                                                             usd_mse_val_1_pso,
                                                                                                             usd_mse_teste_1_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(usd_resultados_1_pso).to_csv('resultados/usd_resultados_1_pso.csv')
pd.DataFrame(usd_resultados_mse_treino_1_pso).to_csv('resultados/usd_resultados_mse_treino_1_pso.csv')
pd.DataFrame(usd_resultados_mse_teste_1_pso).to_csv('resultados/usd_resultados_mse_teste_1_pso.csv')

print('CQSO')
# cqso
tic = time.time()
usd_mse_treino_1_cqso, usd_mse_val_1_cqso, usd_mse_teste_1_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
usd_resultados_1_cqso, usd_resultados_mse_treino_1_cqso, usd_resultados_mse_teste_1_cqso = avaliacao_resultados(usd_mse_treino_1_cqso, 
                                                                                                             usd_mse_val_1_cqso,
                                                                                                             usd_mse_teste_1_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(usd_resultados_1_cqso).to_csv('resultados/usd_resultados_1_cqso.csv')
pd.DataFrame(usd_resultados_mse_treino_1_cqso).to_csv('resultados/usd_resultados_mse_treino_1_cqso.csv')
pd.DataFrame(usd_resultados_mse_teste_1_cqso).to_csv('resultados/usd_resultados_mse_teste_1_cqso.csv')


# ### Cenário II - USD
# 
# * w = 20
# * s = 8
# * f = 100


w = 20 # tamanho da janela
s = 8 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(usd)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)

tic = time.time()
#backprop
print('BACKPROP')
usd_mse_treino_2_backprop, usd_mse_val_2_backprop, usd_mse_teste_2_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
usd_resultados_2_backprop, usd_resultados_mse_treino_2_backprop, usd_resultados_mse_teste_2_backprop = avaliacao_resultados(usd_mse_treino_2_backprop, 
                                                                                                                            usd_mse_val_2_backprop, 
                                                                                                                            usd_mse_teste_2_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(usd_resultados_2_backprop).to_csv('resultados/usd_resultados_2_backprop.csv')
pd.DataFrame(usd_resultados_mse_treino_2_backprop).to_csv('resultados/usd_resultados_mse_treino_2_backprop.csv')
pd.DataFrame(usd_resultados_mse_teste_2_backprop).to_csv('resultados/usd_resultados_mse_teste_2_backprop.csv')

print('PSO')
# pso
tic = time.time()
usd_mse_treino_2_pso, usd_mse_val_2_pso, usd_mse_teste_2_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
usd_resultados_2_pso, usd_resultados_mse_treino_2_pso, usd_resultados_mse_teste_2_pso = avaliacao_resultados(usd_mse_treino_2_pso, 
                                                                                                             usd_mse_val_2_pso,
                                                                                                             usd_mse_teste_2_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(usd_resultados_2_pso).to_csv('resultados/usd_resultados_2_pso.csv')
pd.DataFrame(usd_resultados_mse_treino_2_pso).to_csv('resultados/usd_resultados_mse_treino_2_pso.csv')
pd.DataFrame(usd_resultados_mse_teste_2_pso).to_csv('resultados/usd_resultados_mse_teste_2_pso.csv')

print('CQSO')
# cqso
tic = time.time()
usd_mse_treino_2_cqso, usd_mse_val_2_cqso, usd_mse_teste_2_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
usd_resultados_2_cqso, usd_resultados_mse_treino_2_cqso, usd_resultados_mse_teste_2_cqso = avaliacao_resultados(usd_mse_treino_2_cqso, 
                                                                                                             usd_mse_val_2_cqso,
                                                                                                             usd_mse_teste_2_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(usd_resultados_2_cqso).to_csv('resultados/usd_resultados_2_cqso.csv')
pd.DataFrame(usd_resultados_mse_treino_2_cqso).to_csv('resultados/usd_resultados_mse_treino_2_cqso.csv')
pd.DataFrame(usd_resultados_mse_teste_2_cqso).to_csv('resultados/usd_resultados_mse_teste_2_cqso.csv')


# ### Cenário III - USD
# 
# * w = 20
# * s = 16
# * f = 150

w = 20 # tamanho da janela
s = 16 # tamanho do passo
f = 150 # quantidade de iteracões para a janela
T = int(f/s*(len(usd)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
usd_mse_treino_3_backprop, usd_mse_val_3_backprop, usd_mse_teste_3_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
usd_resultados_3_backprop, usd_resultados_mse_treino_3_backprop, usd_resultados_mse_teste_3_backprop = avaliacao_resultados(usd_mse_treino_3_backprop, 
                                                                                                                            usd_mse_val_3_backprop, 
                                                                                                                            usd_mse_teste_3_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(usd_resultados_3_backprop).to_csv('resultados/usd_resultados_3_backprop.csv')
pd.DataFrame(usd_resultados_mse_treino_3_backprop).to_csv('resultados/usd_resultados_mse_treino_3_backprop.csv')
pd.DataFrame(usd_resultados_mse_teste_3_backprop).to_csv('resultados/usd_resultados_mse_teste_3_backprop.csv')


print('PSO')
# pso
tic = time.time()
usd_mse_treino_3_pso, usd_mse_val_3_pso, usd_mse_teste_3_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
usd_resultados_3_pso, usd_resultados_mse_treino_3_pso, usd_resultados_mse_teste_3_pso = avaliacao_resultados(usd_mse_treino_3_pso, 
                                                                                                             usd_mse_val_3_pso,
                                                                                                             usd_mse_teste_3_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(usd_resultados_3_pso).to_csv('resultados/usd_resultados_3_pso.csv')
pd.DataFrame(usd_resultados_mse_treino_3_pso).to_csv('resultados/usd_resultados_mse_treino_3_pso.csv')
pd.DataFrame(usd_resultados_mse_teste_3_pso).to_csv('resultados/usd_resultados_mse_teste_3_pso.csv')

print('CQSO')
# cqso
tic = time.time()
usd_mse_treino_3_cqso, usd_mse_val_3_cqso, usd_mse_teste_3_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
usd_resultados_3_cqso, usd_resultados_mse_treino_3_cqso, usd_resultados_mse_teste_3_cqso = avaliacao_resultados(usd_mse_treino_3_cqso, 
                                                                                                             usd_mse_val_3_cqso,
                                                                                                             usd_mse_teste_3_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(usd_resultados_3_cqso).to_csv('resultados/usd_resultados_3_cqso.csv')
pd.DataFrame(usd_resultados_mse_treino_3_cqso).to_csv('resultados/usd_resultados_mse_treino_3_cqso.csv')
pd.DataFrame(usd_resultados_mse_teste_3_cqso).to_csv('resultados/usd_resultados_mse_teste_3_cqso.csv')


# ### Cenário IV - USD
# 
# * w = 20
# * s = 20
# * f = 100


w = 20 # tamanho da janela
s = 20 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(usd)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
usd_mse_treino_4_backprop, usd_mse_val_4_backprop, usd_mse_teste_4_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
usd_resultados_4_backprop, usd_resultados_mse_treino_4_backprop, usd_resultados_mse_teste_4_backprop = avaliacao_resultados(usd_mse_treino_4_backprop, 
                                                                                                                            usd_mse_val_4_backprop, 
                                                                                                                            usd_mse_teste_4_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(usd_resultados_4_backprop).to_csv('resultados/usd_resultados_4_backprop.csv')
pd.DataFrame(usd_resultados_mse_treino_4_backprop).to_csv('resultados/usd_resultados_mse_treino_4_backprop.csv')
pd.DataFrame(usd_resultados_mse_teste_4_backprop).to_csv('resultados/usd_resultados_mse_teste_4_backprop.csv')


print('PSO')
# pso
tic = time.time()
usd_mse_treino_4_pso, usd_mse_val_4_pso, usd_mse_teste_4_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
usd_resultados_4_pso, usd_resultados_mse_treino_4_pso, usd_resultados_mse_teste_4_pso = avaliacao_resultados(usd_mse_treino_4_pso, 
                                                                                                             usd_mse_val_4_pso,
                                                                                                             usd_mse_teste_4_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(usd_resultados_4_pso).to_csv('resultados/usd_resultados_4_pso.csv')
pd.DataFrame(usd_resultados_mse_treino_4_pso).to_csv('resultados/usd_resultados_mse_treino_4_pso.csv')
pd.DataFrame(usd_resultados_mse_teste_4_pso).to_csv('resultados/usd_resultados_mse_teste_4_pso.csv')

print('CQSO')
# cqso
tic = time.time()
usd_mse_treino_4_cqso, usd_mse_val_4_cqso, usd_mse_teste_4_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
usd_resultados_4_cqso, usd_resultados_mse_treino_4_cqso, usd_resultados_mse_teste_4_cqso = avaliacao_resultados(usd_mse_treino_4_cqso, 
                                                                                                             usd_mse_val_4_cqso,
                                                                                                             usd_mse_teste_4_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(usd_resultados_4_cqso).to_csv('resultados/usd_resultados_4_cqso.csv')
pd.DataFrame(usd_resultados_mse_treino_4_cqso).to_csv('resultados/usd_resultados_mse_treino_4_cqso.csv')
pd.DataFrame(usd_resultados_mse_teste_4_cqso).to_csv('resultados/usd_resultados_mse_teste_4_cqso.csv')


# ## 6. Hourly internet traffic time series (HIT)
# 
# * 1657 obs.
# * hourly 
# * 19 nov 2004 at 9:30 and 27 jan 2005 at 11:11
# 
# Série horária: 24 inputs
# 

## HIT
hit = pd.read_csv('dados/internet_traffic.csv')
hit = hit['valor']
hit.head()


qtd_inputs = 24
hit_norm = normalizar_serie(hit)
X, y = split_sequence(hit_norm.values, qtd_inputs, 1)


# #### Cenário I - HIT
# 
# * w = 584
# * s = 100
# * f = 50


w = 584 # tamanho da janela
s = 100 # tamanho do passo
f = 50 # quantidade de iteracões para a janela
T = int(f/s*(len(hit)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)



tic = time.time()
#backprop
print('BACKPROP')
hit_mse_treino_1_backprop, hit_mse_val_1_backprop, hit_mse_teste_1_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
hit_resultados_1_backprop, hit_resultados_mse_treino_1_backprop, hit_resultados_mse_teste_1_backprop = avaliacao_resultados(hit_mse_treino_1_backprop, 
                                                                                                                            hit_mse_val_1_backprop, 
                                                                                                                            hit_mse_teste_1_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(hit_resultados_1_backprop).to_csv('resultados/hit_resultados_1_backprop.csv')
pd.DataFrame(hit_resultados_mse_treino_1_backprop).to_csv('resultados/hit_resultados_mse_treino_1_backprop.csv')
pd.DataFrame(hit_resultados_mse_teste_1_backprop).to_csv('resultados/hit_resultados_mse_teste_1_backprop.csv')


print('PSO')
# pso
tic = time.time()
hit_mse_treino_1_pso, hit_mse_val_1_pso, hit_mse_teste_1_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
hit_resultados_1_pso, hit_resultados_mse_treino_1_pso, hit_resultados_mse_teste_1_pso = avaliacao_resultados(hit_mse_treino_1_pso, 
                                                                                                             hit_mse_val_1_pso,
                                                                                                             hit_mse_teste_1_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(hit_resultados_1_pso).to_csv('resultados/hit_resultados_1_pso.csv')
pd.DataFrame(hit_resultados_mse_treino_1_pso).to_csv('resultados/hit_resultados_mse_treino_1_pso.csv')
pd.DataFrame(hit_resultados_mse_teste_1_pso).to_csv('resultados/hit_resultados_mse_teste_1_pso.csv')

print('CQSO')
# cqso
tic = time.time()
hit_mse_treino_1_cqso, hit_mse_val_1_cqso, hit_mse_teste_1_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
hit_resultados_1_cqso, hit_resultados_mse_treino_1_cqso, hit_resultados_mse_teste_1_cqso = avaliacao_resultados(hit_mse_treino_1_cqso, 
                                                                                                             hit_mse_val_1_cqso,
                                                                                                             hit_mse_teste_1_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(hit_resultados_1_cqso).to_csv('resultados/hit_resultados_1_cqso.csv')
pd.DataFrame(hit_resultados_mse_treino_1_cqso).to_csv('resultados/hit_resultados_mse_treino_1_cqso.csv')
pd.DataFrame(hit_resultados_mse_teste_1_cqso).to_csv('resultados/hit_resultados_mse_teste_1_cqso.csv')


# #### Cenário II - HIT
# 
# * w = 584
# * s = 250
# * f = 100
# 


w = 584 # tamanho da janela
s = 250 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(hit)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)



tic = time.time()
#backprop
print('BACKPROP')
hit_mse_treino_2_backprop, hit_mse_val_2_backprop, hit_mse_teste_2_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
hit_resultados_2_backprop, hit_resultados_mse_treino_2_backprop, hit_resultados_mse_teste_2_backprop = avaliacao_resultados(hit_mse_treino_2_backprop, 
                                                                                                                            hit_mse_val_2_backprop, 
                                                                                                                            hit_mse_teste_2_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(hit_resultados_2_backprop).to_csv('resultados/hit_resultados_2_backprop.csv')
pd.DataFrame(hit_resultados_mse_treino_2_backprop).to_csv('resultados/hit_resultados_mse_treino_2_backprop.csv')
pd.DataFrame(hit_resultados_mse_teste_2_backprop).to_csv('resultados/hit_resultados_mse_teste_2_backprop.csv')


print('PSO')
# pso
tic = time.time()
hit_mse_treino_2_pso, hit_mse_val_2_pso, hit_mse_teste_2_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
hit_resultados_2_pso, hit_resultados_mse_treino_2_pso, hit_resultados_mse_teste_2_pso = avaliacao_resultados(hit_mse_treino_2_pso, 
                                                                                                             hit_mse_val_2_pso,
                                                                                                             hit_mse_teste_2_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(hit_resultados_2_pso).to_csv('resultados/hit_resultados_2_pso.csv')
pd.DataFrame(hit_resultados_mse_treino_2_pso).to_csv('resultados/hit_resultados_mse_treino_2_pso.csv')
pd.DataFrame(hit_resultados_mse_teste_2_pso).to_csv('resultados/hit_resultados_mse_teste_2_pso.csv')

print('CQSO')
# cqso
tic = time.time()
hit_mse_treino_2_cqso, hit_mse_val_2_cqso, hit_mse_teste_2_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
hit_resultados_2_cqso, hit_resultados_mse_treino_2_cqso, hit_resultados_mse_teste_2_cqso = avaliacao_resultados(hit_mse_treino_2_cqso, 
                                                                                                             hit_mse_val_2_cqso,
                                                                                                             hit_mse_teste_2_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(hit_resultados_2_cqso).to_csv('resultados/hit_resultados_2_cqso.csv')
pd.DataFrame(hit_resultados_mse_treino_2_cqso).to_csv('resultados/hit_resultados_mse_treino_2_cqso.csv')
pd.DataFrame(hit_resultados_mse_teste_2_cqso).to_csv('resultados/hit_resultados_mse_teste_2_cqso.csv')


# #### Cenário III - HIT
# 
# * w = 584
# * s = 500
# * f = 150
# 
# 


w = 584 # tamanho da janela
s = 500 # tamanho do passo
f = 150 # quantidade de iteracões para a janela
T = int(f/s*(len(hit)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)



tic = time.time()
#backprop
print('BACKPROP')
hit_mse_treino_3_backprop, hit_mse_val_3_backprop, hit_mse_teste_3_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
hit_resultados_3_backprop, hit_resultados_mse_treino_3_backprop, hit_resultados_mse_teste_3_backprop = avaliacao_resultados(hit_mse_treino_3_backprop, 
                                                                                                                            hit_mse_val_3_backprop, 
                                                                                                                            hit_mse_teste_3_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(hit_resultados_3_backprop).to_csv('resultados/hit_resultados_3_backprop.csv')
pd.DataFrame(hit_resultados_mse_treino_3_backprop).to_csv('resultados/hit_resultados_mse_treino_3_backprop.csv')
pd.DataFrame(hit_resultados_mse_teste_3_backprop).to_csv('resultados/hit_resultados_mse_teste_3_backprop.csv')

print('PSO')
# pso
tic = time.time()
hit_mse_treino_3_pso, hit_mse_val_3_pso, hit_mse_teste_3_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
hit_resultados_3_pso, hit_resultados_mse_treino_3_pso, hit_resultados_mse_teste_3_pso = avaliacao_resultados(hit_mse_treino_3_pso, 
                                                                                                             hit_mse_val_3_pso,
                                                                                                             hit_mse_teste_3_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(hit_resultados_3_pso).to_csv('resultados/hit_resultados_3_pso.csv')
pd.DataFrame(hit_resultados_mse_treino_3_pso).to_csv('resultados/hit_resultados_mse_treino_3_pso.csv')
pd.DataFrame(hit_resultados_mse_teste_3_pso).to_csv('resultados/hit_resultados_mse_teste_3_pso.csv')

print('CQSO')
# cqso
tic = time.time()
hit_mse_treino_3_cqso, hit_mse_val_3_cqso, hit_mse_teste_3_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
hit_resultados_3_cqso, hit_resultados_mse_treino_3_cqso, hit_resultados_mse_teste_3_cqso = avaliacao_resultados(hit_mse_treino_3_cqso, 
                                                                                                             hit_mse_val_3_cqso,
                                                                                                             hit_mse_teste_3_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(hit_resultados_3_cqso).to_csv('resultados/hit_resultados_3_cqso.csv')
pd.DataFrame(hit_resultados_mse_treino_3_cqso).to_csv('resultados/hit_resultados_mse_treino_3_cqso.csv')
pd.DataFrame(hit_resultados_mse_teste_3_cqso).to_csv('resultados/hit_resultados_mse_teste_3_cqso.csv')


# #### Cenário IV - HIT
# 
# * w = 584
# * s = 584
# * f = 100
# 



w = 584 # tamanho da janela
s = 584 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(hit)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
hit_mse_treino_4_backprop, hit_mse_val_4_backprop, hit_mse_teste_4_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
hit_resultados_4_backprop, hit_resultados_mse_treino_4_backprop, hit_resultados_mse_teste_4_backprop = avaliacao_resultados(hit_mse_treino_4_backprop, 
                                                                                                                            hit_mse_val_4_backprop, 
                                                                                                                            hit_mse_teste_4_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(hit_resultados_4_backprop).to_csv('resultados/hit_resultados_4_backprop.csv')
pd.DataFrame(hit_resultados_mse_treino_4_backprop).to_csv('resultados/hit_resultados_mse_treino_4_backprop.csv')
pd.DataFrame(hit_resultados_mse_teste_4_backprop).to_csv('resultados/hit_resultados_mse_teste_4_backprop.csv')


print('PSO')
# pso
tic = time.time()
hit_mse_treino_4_pso, hit_mse_val_4_pso, hit_mse_teste_4_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
hit_resultados_4_pso, hit_resultados_mse_treino_4_pso, hit_resultados_mse_teste_4_pso = avaliacao_resultados(hit_mse_treino_4_pso, 
                                                                                                             hit_mse_val_4_pso,
                                                                                                             hit_mse_teste_4_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(hit_resultados_4_pso).to_csv('resultados/hit_resultados_4_pso.csv')
pd.DataFrame(hit_resultados_mse_treino_4_pso).to_csv('resultados/hit_resultados_mse_treino_4_pso.csv')
pd.DataFrame(hit_resultados_mse_teste_4_pso).to_csv('resultados/hit_resultados_mse_teste_4_pso.csv')

print('CQSO')
# cqso
tic = time.time()
hit_mse_treino_4_cqso, hit_mse_val_4_cqso, hit_mse_teste_4_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
hit_resultados_4_cqso, hit_resultados_mse_treino_4_cqso, hit_resultados_mse_teste_4_cqso = avaliacao_resultados(hit_mse_treino_4_cqso, 
                                                                                                             hit_mse_val_4_cqso,
                                                                                                             hit_mse_teste_4_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(hit_resultados_4_cqso).to_csv('resultados/hit_resultados_4_cqso.csv')
pd.DataFrame(hit_resultados_mse_treino_4_cqso).to_csv('resultados/hit_resultados_mse_treino_4_cqso.csv')
pd.DataFrame(hit_resultados_mse_teste_4_cqso).to_csv('resultados/hit_resultados_mse_teste_4_cqso.csv')


# ## 7. Daily minimum temperature time series (DMT)
# 
# * 3650 obs.
# * daily - 1981 to 1990
# 
# Daily time series: 30 inputs

## DMT
dmt = pd.read_csv('dados/daily_temp.csv')
dmt = dmt['valor']
dmt.head()

qtd_inputs = 30
hit_norm = normalizar_serie(hit)
X, y = split_sequence(hit_norm.values, qtd_inputs, 1)


# ### Cenário I - DMT
# 
# * w = 510
# * s = 100
# * f = 50


w = 510 # tamanho da janela
s = 100 # tamanho do passo
f = 50 # quantidade de iteracões para a janela
T = int(f/s*(len(dmt)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)



tic = time.time()
#backprop
print('BACKPROP')
dmt_mse_treino_1_backprop, dmt_mse_val_1_backprop, dmt_mse_teste_1_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
dmt_resultados_1_backprop, dmt_resultados_mse_treino_1_backprop, dmt_resultados_mse_teste_1_backprop = avaliacao_resultados(dmt_mse_treino_1_backprop, 
                                                                                                                            dmt_mse_val_1_backprop, 
                                                                                                                            dmt_mse_teste_1_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(dmt_resultados_1_backprop).to_csv('resultados/dmt_resultados_1_backprop.csv')
pd.DataFrame(dmt_resultados_mse_treino_1_backprop).to_csv('resultados/dmt_resultados_mse_treino_1_backprop.csv')
pd.DataFrame(dmt_resultados_mse_teste_1_backprop).to_csv('resultados/dmt_resultados_mse_teste_1_backprop.csv')


print('PSO')
# pso
tic = time.time()
dmt_mse_treino_1_pso, dmt_mse_val_1_pso, dmt_mse_teste_1_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
dmt_resultados_1_pso, dmt_resultados_mse_treino_1_pso, dmt_resultados_mse_teste_1_pso = avaliacao_resultados(dmt_mse_treino_1_pso, 
                                                                                                             dmt_mse_val_1_pso,
                                                                                                             dmt_mse_teste_1_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(dmt_resultados_1_pso).to_csv('resultados/dmt_resultados_1_pso.csv')
pd.DataFrame(dmt_resultados_mse_treino_1_pso).to_csv('resultados/dmt_resultados_mse_treino_1_pso.csv')
pd.DataFrame(dmt_resultados_mse_teste_1_pso).to_csv('resultados/dmt_resultados_mse_teste_1_pso.csv')

print('CQSO')
# cqso
tic = time.time()
dmt_mse_treino_1_cqso, dmt_mse_val_1_cqso, dmt_mse_teste_1_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
dmt_resultados_1_cqso, dmt_resultados_mse_treino_1_cqso, dmt_resultados_mse_teste_1_cqso = avaliacao_resultados(dmt_mse_treino_1_cqso, 
                                                                                                             dmt_mse_val_1_cqso,
                                                                                                             dmt_mse_teste_1_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(dmt_resultados_1_cqso).to_csv('resultados/dmt_resultados_1_cqso.csv')
pd.DataFrame(dmt_resultados_mse_treino_1_cqso).to_csv('resultados/dmt_resultados_mse_treino_1_cqso.csv')
pd.DataFrame(dmt_resultados_mse_teste_1_cqso).to_csv('resultados/dmt_resultados_mse_teste_1_cqso.csv')


# ### Cenário II - DMT
# 
# * w = 510
# * s = 200
# * f = 100


w = 510 # tamanho da janela
s = 200 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(dmt)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
dmt_mse_treino_2_backprop, dmt_mse_val_2_backprop, dmt_mse_teste_2_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
dmt_resultados_2_backprop, dmt_resultados_mse_treino_2_backprop, dmt_resultados_mse_teste_2_backprop = avaliacao_resultados(dmt_mse_treino_2_backprop, 
                                                                                                                            dmt_mse_val_2_backprop, 
                                                                                                                            dmt_mse_teste_2_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(dmt_resultados_2_backprop).to_csv('resultados/dmt_resultados_2_backprop.csv')
pd.DataFrame(dmt_resultados_mse_treino_2_backprop).to_csv('resultados/dmt_resultados_mse_treino_2_backprop.csv')
pd.DataFrame(dmt_resultados_mse_teste_2_backprop).to_csv('resultados/dmt_resultados_mse_teste_2_backprop.csv')


print('PSO')
# pso
tic = time.time()
dmt_mse_treino_2_pso, dmt_mse_val_2_pso, dmt_mse_teste_2_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
dmt_resultados_2_pso, dmt_resultados_mse_treino_2_pso, dmt_resultados_mse_teste_2_pso = avaliacao_resultados(dmt_mse_treino_2_pso, 
                                                                                                             dmt_mse_val_2_pso,
                                                                                                             dmt_mse_teste_2_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(dmt_resultados_2_pso).to_csv('resultados/dmt_resultados_2_pso.csv')
pd.DataFrame(dmt_resultados_mse_treino_2_pso).to_csv('resultados/dmt_resultados_mse_treino_2_pso.csv')
pd.DataFrame(dmt_resultados_mse_teste_2_pso).to_csv('resultados/dmt_resultados_mse_teste_2_pso.csv')

print('CQSO')
# cqso
tic = time.time()
dmt_mse_treino_2_cqso, dmt_mse_val_2_cqso, dmt_mse_teste_2_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
dmt_resultados_2_cqso, dmt_resultados_mse_treino_2_cqso, dmt_resultados_mse_teste_2_cqso = avaliacao_resultados(dmt_mse_treino_2_cqso, 
                                                                                                             dmt_mse_val_2_cqso,
                                                                                                             dmt_mse_teste_2_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(dmt_resultados_2_cqso).to_csv('resultados/dmt_resultados_2_cqso.csv')
pd.DataFrame(dmt_resultados_mse_treino_2_cqso).to_csv('resultados/dmt_resultados_mse_treino_2_cqso.csv')
pd.DataFrame(dmt_resultados_mse_teste_2_cqso).to_csv('resultados/dmt_resultados_mse_teste_2_cqso.csv')


# ### Cenário III - DMT
# 
# * w = 510
# * s = 400
# * f = 150


w = 510 # tamanho da janela
s = 400 # tamanho do passo
f = 150 # quantidade de iteracões para a janela
T = int(f/s*(len(dmt)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)


tic = time.time()
#backprop
print('BACKPROP')
dmt_mse_treino_3_backprop, dmt_mse_val_3_backprop, dmt_mse_teste_3_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
dmt_resultados_3_backprop, dmt_resultados_mse_treino_3_backprop, dmt_resultados_mse_teste_3_backprop = avaliacao_resultados(dmt_mse_treino_3_backprop, 
                                                                                                                            dmt_mse_val_3_backprop, 
                                                                                                                            dmt_mse_teste_3_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(dmt_resultados_3_backprop).to_csv('resultados/dmt_resultados_3_backprop.csv')
pd.DataFrame(dmt_resultados_mse_treino_3_backprop).to_csv('resultados/dmt_resultados_mse_treino_3_backprop.csv')
pd.DataFrame(dmt_resultados_mse_teste_3_backprop).to_csv('resultados/dmt_resultados_mse_teste_3_backprop.csv')


print('PSO')
# pso
tic = time.time()
dmt_mse_treino_3_pso, dmt_mse_val_3_pso, dmt_mse_teste_3_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
dmt_resultados_3_pso, dmt_resultados_mse_treino_3_pso, dmt_resultados_mse_teste_3_pso = avaliacao_resultados(dmt_mse_treino_3_pso, 
                                                                                                             dmt_mse_val_3_pso,
                                                                                                             dmt_mse_teste_3_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(dmt_resultados_3_pso).to_csv('resultados/dmt_resultados_3_pso.csv')
pd.DataFrame(dmt_resultados_mse_treino_3_pso).to_csv('resultados/dmt_resultados_mse_treino_3_pso.csv')
pd.DataFrame(dmt_resultados_mse_teste_3_pso).to_csv('resultados/dmt_resultados_mse_teste_3_pso.csv')

print('CQSO')
# cqso
tic = time.time()
dmt_mse_treino_3_cqso, dmt_mse_val_3_cqso, dmt_mse_teste_3_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
dmt_resultados_3_cqso, dmt_resultados_mse_treino_3_cqso, dmt_resultados_mse_teste_3_cqso = avaliacao_resultados(dmt_mse_treino_3_cqso, 
                                                                                                             dmt_mse_val_3_cqso,
                                                                                                             dmt_mse_teste_3_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(dmt_resultados_3_cqso).to_csv('resultados/dmt_resultados_3_cqso.csv')
pd.DataFrame(dmt_resultados_mse_treino_3_cqso).to_csv('resultados/dmt_resultados_mse_treino_3_cqso.csv')
pd.DataFrame(dmt_resultados_mse_teste_3_cqso).to_csv('resultados/dmt_resultados_mse_teste_3_cqso.csv')


# ### Cenário IV - DMT
# 
# * w = 510
# * s = 510
# * f = 100


w = 510 # tamanho da janela
s = 510 # tamanho do passo
f = 100 # quantidade de iteracões para a janela
T = int(f/s*(len(dmt)-w)+f)
print(T)
quantidade_janelas = int((len(y) - w)/s)



tic = time.time()
#backprop
print('BACKPROP')
dmt_mse_treino_4_backprop, dmt_mse_val_4_backprop, dmt_mse_teste_4_backprop = cenarios_execucoes(X, y, w, s, f, modelo = nn_model2, 
                                                                                                 perc_treino=0.54, perc_val=0.24)
dmt_resultados_4_backprop, dmt_resultados_mse_treino_4_backprop, dmt_resultados_mse_teste_4_backprop = avaliacao_resultados(dmt_mse_treino_4_backprop, 
                                                                                                                            dmt_mse_val_4_backprop, 
                                                                                                                            dmt_mse_teste_4_backprop, 
                                                                                                                            f, quantidade_janelas, 
                                                                                                                            execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o backprop', (tac - tic))

pd.DataFrame(dmt_resultados_4_backprop).to_csv('resultados/dmt_resultados_4_backprop.csv')
pd.DataFrame(dmt_resultados_mse_treino_4_backprop).to_csv('resultados/dmt_resultados_mse_treino_4_backprop.csv')
pd.DataFrame(dmt_resultados_mse_teste_4_backprop).to_csv('resultados/dmt_resultados_mse_teste_4_backprop.csv')


print('PSO')
# pso
tic = time.time()
dmt_mse_treino_4_pso, dmt_mse_val_4_pso, dmt_mse_teste_4_pso = cenarios_execucoes(X, y, w, s, f,modelo = nn_model_pso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
dmt_resultados_4_pso, dmt_resultados_mse_treino_4_pso, dmt_resultados_mse_teste_4_pso = avaliacao_resultados(dmt_mse_treino_4_pso, 
                                                                                                             dmt_mse_val_4_pso,
                                                                                                             dmt_mse_teste_4_pso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o PSO', (tac - tic))

pd.DataFrame(dmt_resultados_4_pso).to_csv('resultados/dmt_resultados_4_pso.csv')
pd.DataFrame(dmt_resultados_mse_treino_4_pso).to_csv('resultados/dmt_resultados_mse_treino_4_pso.csv')
pd.DataFrame(dmt_resultados_mse_teste_4_pso).to_csv('resultados/dmt_resultados_mse_teste_4_pso.csv')

print('CQSO')
# cqso
tic = time.time()
dmt_mse_treino_4_cqso, dmt_mse_val_4_cqso, dmt_mse_teste_4_cqso = cenarios_execucoes(X, y, w, s, f, modelo = nn_model_cqso_todos, 
                                                                                  perc_treino=0.54, perc_val=0.24)
dmt_resultados_4_cqso, dmt_resultados_mse_treino_4_cqso, dmt_resultados_mse_teste_4_cqso = avaliacao_resultados(dmt_mse_treino_4_cqso, 
                                                                                                             dmt_mse_val_4_cqso,
                                                                                                             dmt_mse_teste_4_cqso, 
                                                                                                             f, quantidade_janelas, 
                                                                                                             execucoes = 30) 
tac = time.time()
print('Tempo de execucao para o CQSO', (tac - tic))

pd.DataFrame(dmt_resultados_4_cqso).to_csv('resultados/dmt_resultados_4_cqso.csv')
pd.DataFrame(dmt_resultados_mse_treino_4_cqso).to_csv('resultados/dmt_resultados_mse_treino_4_cqso.csv')
pd.DataFrame(dmt_resultados_mse_teste_4_cqso).to_csv('resultados/dmt_resultados_mse_teste_4_cqso.csv')



