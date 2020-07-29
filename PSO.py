####### Algoritmo PSO ########

# Importando as bibliotecas básicas
import numpy as np

# Criar a função do PSO
def PSO(fun, *args, qtd_particulas, atributos_dim, min, max, seed = np.random.seed(1), max_epoch, w, c1, c2):
    '''
        Função do Algoritmo SWARM PSO. 
        Inputs:
        - fun_opt: Função de fitness a ser otimizada
        - qtd_particulas: Quantidade de partículas
        - atributos_dim: Dimensão do Vetor de atributos 
        - min: intervalo inferior do domínio da função  
        - max: intervalo superior do domínio da função
        - seed: por padrão np.random.seed(1)
        - w: inércia 
        - c1: influência do pbest (termo cognitivo)
        - c2: influência do gbest (termo do aprendizado social)
    '''

    # inicializar as partículas em posições aleatórias
    particulas = np.random.uniform(low = min, high = max, size = (qtd_particulas, atributos_dim))
    #print('Partículas: \n', particulas)

    # inicializar a velocidade
    #velocidade = np.random.uniform(low = min, high = max, size = (qtd_particulas, atributos_dim))
    velocidade = np.zeros((qtd_particulas, atributos_dim))

    # inicializar o pbest em zero
    #pbest = np.ones((qtd_particulas, atributos_dim))
    pbest = np.zeros((qtd_particulas,atributos_dim))

    gbest_value = np.inf
    gbest = 0

    # Extrair a posição do gbest 
    for z in np.arange(qtd_particulas):
        new_gbest_value = fun(particulas[z,:])
        if new_gbest_value < gbest_value:
            gbest = z
    
    gbest_value = particulas[gbest,:]
    #print('Valor da função no gbest:\n', gbest_value)

    funcao_iteracao = np.zeros(max_epoch)
    media = np.zeros(max_epoch)
    desvio_pad = np.zeros(max_epoch)

    for k in np.arange(max_epoch):
    #for k in np.arange(max_epoch):    
        #print('epoch n.:', k)
        #print('\n')
    # Iterar para atualizar o pbest e gbest para cada partrícula
        for j in np.arange(qtd_particulas):
            if fun(particulas[j,:]) < fun(pbest[j,:]):
                pbest[j,:] = particulas[j,:]

            if fun(particulas[j,:]) < fun(particulas[gbest, :]):
                gbest = j
                gbest_value = fun(particulas[gbest, :])
            
            # Iteração para atualizar as posições das partículas
        for i in np.arange(qtd_particulas):
            r1, r2 = np.random.rand(), np.random.rand()
            velocidade[i, :] = w * velocidade[i, :] + c1 * r1 * (pbest[i, :] - particulas[i, :]) + c2 * r2 * (particulas[gbest, :] - particulas[i, :])
            particulas[i, :] = particulas[i, :] + velocidade[i, :]

            # garantir os limites
            for d in np.arange(atributos_dim):
                if particulas[i, d] < min:
                    particiulas[i, d] = min
                elif particulas[i, d] > max:
                    particulas[i, d] = max


        funcao_iteracao[k] = fun(particulas[gbest, :])

        vetor_fitness = np.zeros(qtd_particulas)

        for par in np.arange(qtd_particulas):
            vetor_fitness[par] = fun(particulas[par,:])
        
        media[k] = vetor_fitness.mean()
        desvio_pad[k] = vetor_fitness.std()

    return particulas, gbest, funcao_iteracao, media, desvio_pad


