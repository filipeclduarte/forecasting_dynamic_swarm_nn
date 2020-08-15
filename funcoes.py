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
def cenarios_execucoes(X, y, w, s, f, modelo, perc_treino, perc_val,qtd_execucoes = 30):
    
    # gerando os cenários dinâmicos
    X_I = cenarios_dinamicos(X, w, s)
    y_I = cenarios_dinamicos(y, w, s)
 
    # calculando a quantidade de iterações
    T = int(f/s*(len(y)-w)+f)
    
    neuronios = np.arange(2, 26)
    
    mse_treino = np.zeros((qtd_execucoes, len(neuronios),len(y_I) * f))
    mse_val = np.zeros((qtd_execucoes, len(neuronios), len(y_I) * f))
    mse_teste = np.zeros((qtd_execucoes, len(neuronios),len(y_I) * f))

    execucoes = np.arange(qtd_execucoes)

    for execucao in execucoes:
        print('Execução: ', execucao)

        # Neuronios
        for j,z in zip(neuronios, np.arange(len(neuronios))):
            
            parameters, mse_treino_lista_temp, mse_val_lista_temp, mse_teste_lista_temp = modelo(X_I, y_I, n_h = j, num_iteracoes = f, perc_treino=perc_treino, perc_val=perc_val)

            # salvar lista com os mse de treino para todas as iterações
            mse_treino[execucao, z,:] = np.array(mse_treino_lista_temp)
            # salvar lista com os mse de validacao para todas as iteracoes
            mse_val[execucao, z,:] = np.array(mse_val_lista_temp)
            # salvar lista com os mse de teste para todas as iterações
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
    
    return resultados, mse_treino, mse_teste











np.random.seed(0)
def CQSO(problem, dimensions, var_min, var_max, n_iterations, n_sub_swarms,
         n_particles, w, c1, c2, neutral_p, rcloud, show_iter):
  """ CQSO algorithm """
  # Initialization
  # np.seterr(over='ignore')
  # Divide the dimensions per subswarm
  num, div = dimensions, n_sub_swarms
  dimensions_list = [num // div + (1 if x < num % div else 0)  for x in range (div)]
  if not dimensions % n_sub_swarms == 0:
    print("We can't continue, the number of dimensions isn't divisible by the number of subswarms")
    return False
  # Initialization
  context_vector = np.empty(n_sub_swarms, dtype=object)
  ## Create a multiswarm and his velocities
  multi_swarm_vector = np.empty((n_sub_swarms,n_particles), dtype=object)
  velocity_vector = np.empty((n_sub_swarms,n_particles), dtype=object)
  ### Change None values for random numbers
  for i_subswarm in range(n_sub_swarms):
    context_vector[i_subswarm] = np.random.uniform(
          var_min,var_max,(dimensions_list[i_subswarm]))
    for i_particle in range(n_particles):
      multi_swarm_vector[i_subswarm][i_particle] = np.random.uniform(
          var_min,var_max,(dimensions_list[i_subswarm]))
      velocity_vector[i_subswarm][i_particle] = np.zeros(dimensions_list[i_subswarm])
  ## Create fitness for pbest and gbest
  gbest = np.copy(multi_swarm_vector[0][0])
  pbest = np.copy(multi_swarm_vector[0][0])
  sub_swarm_pbest = np.copy(context_vector)
  best_pfitness = problem(np.concatenate(context_vector))
  best_gfitness = problem(np.concatenate(context_vector))
  iteration = 0
  result_list = []

  while iteration < n_iterations:
    # Iterations
    # for sub_swarm in multi_swarm_vector:
    for i_sub_swarm in range(n_sub_swarms):
      # for particle in sub_swarm:
      for i_particle in range(n_particles):
        # Calculate the fitness
        # Vamos calcular o fitness da particula dentro do vetor contextos
        context_copy = np.copy(context_vector)
        particle = multi_swarm_vector[i_sub_swarm][i_particle]
        context_copy[i_sub_swarm] = particle
        fitness_candidate = problem(np.concatenate(context_copy))
        if fitness_candidate < best_pfitness:
          # Se o fitness da nova particula for melhor ela vira o pbest
          pbest = np.copy(multi_swarm_vector[i_sub_swarm][i_particle])
          best_pfitness = fitness_candidate
          sub_swarm_pbest = np.copy(context_copy)
        # Feito o pbest devemos atualizar as posicoes das particulas
        if i_particle <= (neutral_p - 1):
          # Atualiza como PSO vanilla
          new_velocity = (w * velocity_vector[i_sub_swarm][i_particle]) + \
          ((c1 * random.random()) * (pbest - multi_swarm_vector[i_sub_swarm][i_particle])) + \
          ((c2 * random.random()) * (gbest - multi_swarm_vector[i_sub_swarm][i_particle]))
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
          new_position[index] = np.max([var_min, value])
          new_position[index] = np.min([var_max, new_position[index]])
        multi_swarm_vector[i_sub_swarm][i_particle] = new_position
      # Visto todas as particulas do subswarm eu comparo o gbest
      if best_pfitness < best_gfitness:
        gbest = np.copy(pbest)
        best_gfitness = best_pfitness
        context_vector = np.copy(sub_swarm_pbest)
    result_list.append(best_gfitness)
    iteration += 1

  return result_list


#### PSO para otimizar todos os parâmetros de uma só vez 
    
def PSO_todos(X, parameters_stacked, best_cost, fun, A2, Y, parameters, qtd_particulas, atributos_dim, min_i, max_i, 
                max_epoch, perc_treino, perc_val, w_in=0.7, w_fim = 0.2, c1=1.496, c2=1.496):
    '''
        Função do Algoritmo SWARM PSO. 
        Inputs:
        - X: Input data - windows splited
        - fun_opt: Função de fitness a ser otimizada
        - qtd_particulas: Quantidade de partículas
        - atributos_dim: Dimensão do Vetor de atributos 
        - min: intervalo inferior do domínio da função  
        - max: intervalo superior do domínio da função
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

    # inicializar as partículas em posições aleatórias
    particulas = np.random.uniform(low = min_i, high = max_i, size = (qtd_particulas, atributos_dim_sum))

    # inicializar a velocidade
    velocidade = np.zeros((qtd_particulas, atributos_dim_sum))

    # inicializar o pbest em zero
    pbest = np.zeros((qtd_particulas,atributos_dim_sum))

    gbest_value = best_cost

    gbest = 0
    
    parameters_gbest_dict = parameters.copy()
    parameters_dict = parameters.copy()

    # Extrair a posição do gbest 
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
        #print('Iteração: ', k)
        # Atualização do decaimento do peso
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
                    #print('Parar o algoritmo pois o gbest não melhorou')
                        gbest_value = fun(A2_part, Y_treino.T, parameters_temp_dict)
                        gbest = j
                        parameters_gbest_dict = parameters_temp_dict
                        break
        
                    gbest = j
                    gbest_value = fun(A2_part, Y_treino.T, parameters_temp_dict)
                    parameters_gbest_dict = parameters_temp_dict
                                      
         # Iteração para atualizar as posições das partículas
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
            mse_tv = fun(A2_gbest_tv, Y_tv, parameters_temp_dict)
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
    compute_cost2 - função a ser minimizada, neste caso a função de custo
    A2 - previsão feita pelo modelo
    Y - rótulo 

    Retorna:
    parameters - parâmetros atualizados a partir do PSO
    '''

    # Extrair os parâmetros do dicionário para calcular a dimensão total e para criar o array colunas
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Extrair a dimensão total 
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

    parameters_pso, treino_mse, val_mse, teste_mse
Tempo de execução para o backprop = PSO_todos(X, parameters_stacked, 
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
    print_cost -- se True, mostra o custo a cada 1000 iterações
    
    Retorna:
    parameters -- parâmetros aprendidos pelo pso. Eles podem ser utilizados para fazer previsões (predict).
    """
    
    n_x = layer_sizes2(X[0].T, Y[0].T)[0]
    n_y = layer_sizes2(X[0].T, Y[0].T)[2]
    
    # Inicialização dos parâmetros
    parameters = initialize_parameters2(n_x, n_h, n_y)
    
    A2, _ = forward_propagation2(X[0].T, parameters)

    best_cost = compute_cost2(A2, Y[0].T, parameters)
    
    # Atualização dos parâmetros pelo gradiente descendente. Inputs: "parameters, compute_cost2, A2, Y". Outputs: "parameters".
    parameters, treino_mse, val_mse, teste_mse = update_parameters_pso_todos(X=X, parameters=parameters, best_cost=best_cost,compute_cost2=compute_cost2, 
                                                                             A2=A2, Y=Y, num_iteracoes=num_iteracoes, perc_treino=perc_treino, perc_val=perc_val)

    
    return parameters, treino_mse, val_mse, teste_mse