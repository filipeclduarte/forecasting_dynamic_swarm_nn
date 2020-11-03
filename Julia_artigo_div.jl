using Distributions
using CSV
using Plots
using DataFrames
using Random
using ArgParse
using Parameters

Random.seed!(123);

function init_params(n_in::Int64, n_h::Int64)
    W1 = rand(Uniform(-1/sqrt(n_h+n_in),1/sqrt(n_h+n_in)),n_h, n_in)
    b1 = rand(Uniform(-1/sqrt(n_h),1/sqrt(n_h)),n_h)
    W2 = rand(Uniform(-1/sqrt(1+n_h),1/sqrt(1+n_h)), 1, n_h)
    b2 = rand(Uniform(-1,1), 1, 1)

    return (W1=W1, b1=b1, W2=W2, b2=b2)
end

function forward_prop(x, params)
    # W1 = params[1]
    # b1 = params[2]
    # W2 = params[3]
    # b2 = params[4]
    
    @unpack W1, b1, W2, b2 = params

    linear1 = W1 * x .+ b1
    linear2 = W2 * linear1 .+ b2
    out = 1.7159 * tanh.(2/3 * linear2)
    
    return out

end

function compute_cost(A2, y)
    m = size(y)[2]
    erro = A2 .- y
    cost = 1/m * sum(erro .^ 2)
    return cost
end

function params_array(params)
    
    @unpack W1, b1, W2, b2 = params

    # lista_params = hcat(reshape(W1, 1,prod(size(W1))), reshape(b1, 1,prod(size(b1))), 
    #   reshape(W2, 1,prod(size(W2))), reshape(b2, 1,prod(size(b2))))

    lista_params = [reshape(W1, 1, prod_size(W1)) reshape(b1, 1,prod_size(b1)) reshape(W2, 1,prod_size(W2)) reshape(b2, 1,prod_size(b2))]
    
    return lista_params
end

# função para calcular tamanho da matriz
prod_size(x) = prod(size(x))

function params_reshape(lista_params, params)

    @unpack W1, b1, W2, b2 = params

    w1_dim_tot = prod_size(W1)
    b1_dim_tot = prod_size(b1)
    w2_dim_tot = prod_size(W2)
    b2_dim_tot = prod_size(b2)

    W1_reshaped = reshape(lista_params[1:w1_dim_tot], size(W1))
    id = w1_dim_tot

    b1_reshaped = reshape(lista_params[id+1:id+b1_dim_tot], size(b1))
    id += b1_dim_tot
    
    W2_reshaped = reshape(lista_params[id+1:id+w2_dim_tot], size(W2))
    id += w2_dim_tot
    
    b2_reshaped = reshape(lista_params[id+1:id+b2_dim_tot], size(b2))
    
    return (W1=W1_reshaped, b1=b1_reshaped, W2=W2_reshaped, b2=b2_reshaped)
    
end

# normalizar serie
function normalizar(serie)
    maxi = maximum(serie)
    mini = minimum(serie)
    
    y_temp = 2 * ((serie .- mini) / (maxi - mini)) .- 1
    
    y = y_temp / sqrt(size(serie)[1])
    
    return y
end

function desnormalizar(serie_norm, serie)
    maxi = maximum(serie)
    mini = maximum(serie)
    
    serie_temp = serie_norm * sqrt(size(serie)[1])
    serie_temp2 = (serie_temp .+ 1) / 2
    serie_temp3 = serie_temp2 * ((maxi - mini) + mini)
    return serie_temp3 
end

function split_sequence(serie, n_steps_in::Int64)
    
    len = size(serie)[1]
    max_iter = len - n_steps_in
    
    seq_x = Array{Float64, 2}(undef, max_iter, n_steps_in)
    seq_y = Array{Float64, 1}(undef, max_iter)
    
    for i in 1:len-n_steps_in
        idx = i + n_steps_in - 1
        out_idx = idx + 1
        if out_idx > len
            println(i)
            println("len = ", len)
            println("out_idx = ", out_idx)
            println("Out_idx > len")
            break
        end
        seq_x[i,:] = serie[i:idx]
        seq_y[i] = serie[out_idx]
    end
    return seq_x, seq_y
end

function divisao_dados_temporais(X, y, perc_treino::Float64, perc_val::Float64)
    tam_treino = convert(Int64, floor(perc_treino * size(y)[1]))
    
    if perc_val > 0
        tam_val = convert(Int64, floor(perc_val * size(y)[1]))
        X_treino = X[1:tam_treino,:]
        y_treino = y[1:tam_treino]
        
        X_val = X[tam_treino+1:tam_treino+1+tam_val,:]
        y_val = y[tam_treino+1:tam_treino+1+tam_val]
        
        X_teste = X[(tam_treino+tam_val+1):end,:]
        y_teste = y[(tam_treino+tam_val+1):end]
        
        return X_treino, y_treino, X_teste, y_teste,  X_val, y_val
        
    else
        X_treino = X[1:tam_treino,:]
        y_treino = y[1:tam_treino]
        
        X_teste = X[tam_treino+1:end,:]
        y_teste = y[tam_treino+1:end]
        return X_treino, y_treino, X_teste, y_teste
    end
end

function PSO(X, y, params, n_particles::Int64, max_iter::Int64, LB::Float64, UB::Float64, perc_treino::Float64, perc_val::Float64)
    
    mse_treino = Array{Float64, 1}(undef, size(y)[2]*max_iter)
    mse_val = similar(mse_treino)
    mse_teste = similar(mse_treino)

    particles = rand(n_particles, n_particles)
    velocity = zeros(n_particles, n_particles)
    pbest = copy(particles)
    gbest = pbest[1,:]
    
    parametros = params_reshape(gbest, params)

    # y_pred = forward_prop(X[:,:,1]', parametros)
    # gbest_value = compute_cost(y_pred, y[:,1]')    
    gbest_value = Inf
    fitness_value = Array{Float64, 1}(undef, n_particles)

    for i in eachindex(fitness_value)
        parametros = params_reshape(particles[i,:], params)
        y_pred = forward_prop(X[:,:,1]', parametros)
        fitness_value[i] = compute_cost(y_pred, y[:,1]')
        if fitness_value[i] < gbest_value
            gbest_value = fitness_value[i]
        end
    end

    # salvar o melhor valor
    
    wmax = 0.9
    wmin = 0.4
    c1 = 1.5
    c2 = 1.5
    
    iteracao = 1
    for janela in 1:size(y, 2)
        
        X_treino, Y_treino, X_teste, Y_teste, X_val, Y_val = divisao_dados_temporais(X[:,:,janela], y[:,janela], perc_treino, perc_val)
        
        X_tv = [X_treino; X_val]'
        Y_tv = [Y_treino; Y_val]'
        # X_tv = vcat(X_treino, X_val)'
        # Y_tv = vcat(Y_treino, Y_val)'

        for k in 1:max_iter
            w=wmax-(wmax-wmin)*k/max_iter
        

            for i in 1:n_particles
                for j in 1:n_particles
                    velocity[i, j] = w*velocity[i,j] + c1*rand()*(pbest[i,j] - particles[i,j]) + c2*rand()*(gbest[j] - particles[i,j])
                end
            end

            # update pso position
            for i in 1:n_particles
                for j in 1:n_particles
                    particles[i,j] = particles[i,j] + velocity[i,j]
                end
            end

            # handling boundary violations 
            for i in 1:n_particles
                for j in 1:n_particles
                    if particles[i,j]<LB 
                        particles[i,j]=LB 
                    elseif particles[i,j]>UB 
                        particles[i,j]=UB 
                    end
                end 
            end 

            # evaluating fitness
            for i in 1:n_particles
                parametros = params_reshape(particles[i,:], params)
                y_pred = forward_prop(X_treino', parametros)
                fitness_value[i] = compute_cost(y_pred, Y_treino')
            end

            # updating pbest and fitness
            for i in 1:n_particles
                parametros_pbest = params_reshape(pbest[i, :], params)
                y_pred_pbest = forward_prop(X_treino', parametros_pbest)
                if fitness_value[i] < compute_cost(y_pred_pbest, Y_treino')
                    pbest[i] = particles[i]
                end
            end

            # updating gbest 
            for i in 1:n_particles    
                if fitness_value[i] < gbest_value
                    gbest_value = fitness_value[i]
                    gbest = particles[i,:]
                end
            end
    
            parametros_gbest = params_reshape(gbest, params)
            # treino e validacao mse
            A2_gbest_tv = forward_prop(X_tv, parametros_gbest)
            mse_tv = compute_cost(A2_gbest_tv, Y_tv)
            mse_treino[iteracao] = mse_tv

            # validacao mse
            A2_gbest_v = forward_prop(X_val', parametros_gbest)
            mse_v = compute_cost(A2_gbest_v, Y_val')
            mse_val[iteracao] = mse_v

            # teste
            A2_gbest_t = forward_prop(X_teste', parametros_gbest)
            mse_t = compute_cost(A2_gbest_t, Y_teste')
            mse_teste[iteracao] = mse_t
            
            iteracao += 1
        end
    end
    return mse_treino, mse_val, mse_teste
end

function n_model_pso(X, y, n_h::Int64, num_iteracoes::Int64, perc_treino::Float64, perc_val::Float64)
    
    n_in = size(X, 2)
    parametros = init_params(n_in, n_h)
    
#     best_cost = compute_cost(A2, y')
#     W1 = parametros[1]
#     b1 = parametros[2]
#     W2 = parametros[3]
#     b2 = parametros[4]

    @unpack W1, b1, W2, b2 = parametros

    dim_list = map(prod_size, parametros)
    # dim_list = [prod_size(W1), prod_size(b1), prod_size(W2), prod_size(b2)]
    qtd_particulas_dim = sum(dim_list)
    # qtd_particulas_dim = convert(Int64,(size(W1)[2] + 1) * size(W1)[1] + (size(W1)[1] + 1) * size(W2)[1])

    mse_treino, mse_val, mse_teste = PSO(X, y, parametros, qtd_particulas_dim, num_iteracoes, -1.0, 1.0, perc_treino, perc_val)
    
    return mse_treino, mse_val, mse_teste
end


# janelamento para cenários dinâmicos
function cenarios_dinamicos(serie, window_size::Int64, step_size::Int64)
    
    if ndims(serie) == 1
        w = window_size
        s = step_size
        t = size(serie)[1]

        i_max = convert(Int64,floor((t-w)/s))

        s_temp = serie[(1*s):((1*s)+w-1)]

        cenarios = zeros(0)

        append!(cenarios, s_temp)

        for i in 2:i_max
            s_temp = serie[(i*s):((i*s)+w-1)]
            cenarios = hcat(cenarios, s_temp)
        end
        return cenarios
    else
        w = window_size
        s = step_size
        t = size(serie)[1]

        i_max = convert(Int64,floor((t-w)/s))

        s_temp = serie[(1*s):((1*s)+w-1), :]

        cenarios = zeros(w, size(serie)[2], i_max)

        cenarios[:,:, 1] = s_temp

        for i in 2:i_max
            s_temp = serie[(i*s):((i*s)+w-1),:]
            cenarios[:, :, i] = s_temp
        end
        return cenarios
    end
end

# Criando cenários
function cenarios_execucoes_pso(X, y, w, s, f::Int64, perc_treino::Float64, perc_val::Float64,qtd_execucoes::Int64)
    
    # gerando os cenários dinâmicos
    X_I = cenarios_dinamicos(X, w, s)
    y_I = cenarios_dinamicos(y, w, s)
 
    # calculando a quantidade de iterações
    T = f * size(y_I)[2] 

    println("Quantidade de iterações: ", T)
    
    neuronios = 2:25
    
    mse_treino = Array{Float64, 3}(undef, qtd_execucoes, size(neuronios)[1], size(y_I)[2] * f)
    mse_val = similar(mse_treino)
    mse_teste = similar(mse_treino)
    
    execucoes = 1:qtd_execucoes

    for execucao in eachindex(execucoes)

        println("Execução: ", execucao)
        
        # Neuronios
        for (j,z) in zip(neuronios, eachindex(neuronios))
            
            println("Neurônios: ", j)
            
            mse_treino_lista_temp, mse_val_lista_temp, mse_teste_lista_temp = n_model_pso(X_I, y_I, j, f, perc_treino, perc_val)

            # salvar lista com os mse de treino para todas as iterações
            mse_treino[execucao, z,:] = mse_treino_lista_temp
            # salvar lista com os mse de validacao para todas as iteracoes
            mse_val[execucao, z,:] = mse_val_lista_temp
            # salvar lista com os mse de teste para todas as iterações
            mse_teste[execucao, z,:] = mse_teste_lista_temp

        end
    end
    return mse_treino, mse_val, mse_teste
end
            

function avaliacao_resultados(mse_treino_cenarios, mse_val_cenarios, mse_teste_cenarios, f, quantidade_janelas, execucoes)
    
    mse_treino = Array{Float64, 2}(undef, execucoes, quantidade_janelas*f)
    mse_teste = similar(mse_treino)
    
    for ex in 1:execucoes
        id_neuronios = Array{Int64, 1}(undef, quantidade_janelas)
        for janela in 1:quantidade_janelas
            id_neuronios[janela] = findmin(mse_val_cenarios[ex, :, f*janela])[2]
        end
        mse_treino[ex, 1:f] = mse_treino_cenarios[ex, id_neuronios[1], 1:f]
        mse_teste[ex, 1:f] = mse_teste_cenarios[ex, id_neuronios[1], 1:f]
        
        for jan in 1:quantidade_janelas
            if jan == quantidade_janelas
                mse_treino[ex, f*jan-f+1:f*jan] =  mse_treino_cenarios[ex, id_neuronios[jan], f*jan-f+1:f*jan]
                mse_teste[ex, f*jan-f+1:f*jan] = mse_teste_cenarios[ex, id_neuronios[jan], f*jan-f+1:f*jan]
            else
                mse_treino[ex, f*jan+1:f*jan+f] = mse_treino_cenarios[ex, id_neuronios[jan], f*jan+1:f*jan+f]
                mse_teste[ex, f*jan+1:f*jan+f] = mse_teste_cenarios[ex, id_neuronios[jan], f*jan+1:f*jan+f]
            end
        end
    end
        
    qtd_iteracoes = size(mse_treino)[2]
    
    te = sum(mse_treino, dims = 2)./qtd_iteracoes
    ge = sum(mse_teste, dims = 2)./qtd_iteracoes

    gf = ge./te

    te_medio = mean(te)
    te_std = std(te)

    ge_medio = mean(ge)
    ge_std = std(ge)

    gf_medio = mean(gf)
    gf_std = std(ge)

    println("Te médio: ", te_medio)
    println("TE desvio: ", te_std)
    println("GE medio: ", ge_medio)
    println("GE desvio: ", ge_std)
    println("GF medio: ", gf_medio)
    println("GF desvio: ", gf_std)

    resultados = [te_medio, te_std, ge_medio, ge_std, gf_medio, gf_std]

    return resultados, mse_treino, mse_teste

end  

function CQSO(X, y, params, n_particles::Int64, max_iter::Int64, LB::Float64, UB::Float64, perc_treino::Float64, perc_val::Float64, neutral_p::Int64, rcloud::Float64)
    
    
    mse_treino = Array{Float64, 1}(undef, size(y)[2]*max_iter)
    mse_val = similar(mse_treino)
    mse_teste = similar(mse_treino)

    n = copy(n_particles)
    # n = sum(dim)
    a = zeros(Int64, n)

    for i in 2:n
        if n%i == 0
            a[i] = i
        end
    end

    n_sub_swarms = sort(a[a.!=0])[1]

    # divide the dimensions per subswarm
    num = copy(n)
    div = copy(n_sub_swarms)

    dimensions_list = zeros(Int64, div)

    for x in 1:div
        if x < num % div
            dimensions_list[x] = 1 + (num ÷ div)
        else
            dimensions_list[x] = 0 + (num ÷ div)
        end
    end

    context_vector = Array{Float64, 2}(undef, n_sub_swarms, n)
    
    # Create a multiswarm and his velocities
    multi_swarm_vector = Array{Float64, 3}(undef, n_sub_swarms, n_particles, n)
    # multi_swarm_vector = zeros(n_sub_swarms, n_particles, n)
    velocity_vector = zeros(n_sub_swarms, n_particles, n)
    
    for i_subswarm in 1:n_sub_swarms
        context_vector[i_subswarm, :] = rand(Uniform(-1.0, 1.0), n)
        for i_particle in 1:n_particles
            multi_swarm_vector[i_subswarm, i_particle, :] = rand(Uniform(-1.0, 1.0), n)
        end
    end

    gbest = multi_swarm_vector[1,1,:]
    pbest = multi_swarm_vector[1,1,:]

    #inicializando arrays 
    new_velocity= similar(gbest)
    new_position = similar(gbest)
    uniform = similar(gbest)
    normal = similar(gbest)
    left_size_form = similar(gbest)
    right_size_form = similar(gbest)


    sub_swarm_pbest = context_vector[1,:]
    # parameters = params

    parametros = params_reshape(sub_swarm_pbest, params)
    
    y_pred = forward_prop(X[:,:,1]', parametros)
    gbest_value = compute_cost(y_pred, y[:, 1]')
    # compute pbest value
    pbest_value = copy(gbest_value)
    # copiando os parametros iniciais do sub_swarm_pbest
    parametros_gbest = context_vector[1,:]



    wmax = 0.9
    wmin = 0.4
    c1 = 1.5
    c2 = 1.5

    
    it_idx = 1

    for janela in 1:size(y, 2)
    

        X_treino, Y_treino, X_teste, Y_teste, X_val, Y_val = divisao_dados_temporais(X[:,:, janela], y[:, janela], perc_treino, perc_val)

        X_tv = [X_treino; X_val]'
        Y_tv = [Y_treino; Y_val]'
        # X_tv = vcat(X_treino, X_val)'
        # Y_tv = vcat(Y_treino, Y_val)'

        # Iterações
        # Para cada sub_swarm em multi_swarm_vector
        iteration = 1
        while iteration <= max_iter

            w=wmax-(wmax-wmin)*iteration/max_iter

            # for particula in sub_swarm
            for i_sub_swarm in 1:n_sub_swarms

                for i_particle in 1:n_particles

                    # Calcular o fitness 
                    context_copy = copy(context_vector)
                    particle = multi_swarm_vector[i_sub_swarm,i_particle,:]
                    context_copy[i_sub_swarm, :] = particle

                    parameters_temp = params_reshape(context_copy[i_sub_swarm, :], params)
                    
                    A2_part = forward_prop(X_treino', parameters_temp)
                    fitness_candidate = compute_cost(A2_part, Y_treino')
                    
                    if fitness_candidate < pbest_value
                    # se o fitness da nova particula for melhor, ela vira o pbest
                        pbest = multi_swarm_vector[i_sub_swarm, i_particle, :]
                        pbest_value = copy(fitness_candidate)
                        sub_swarm_pbest = context_copy[i_sub_swarm, :]
                         # feito o pbest devemos atualizar as posições das particulas
                    end

                    if i_particle <= (neutral_p - 1)  

                        for j in 1:n_particles
                        # atualiza como PSO vanilla
                            new_velocity[j] = (w * velocity_vector[i_sub_swarm, i_particle, j]) + 
                            ((c1 * rand()) * (pbest[j] - multi_swarm_vector[i_sub_swarm, i_particle, j])) +
                            ((c2 * rand()) * (gbest[j] - multi_swarm_vector[i_sub_swarm, i_particle, j]))

                            new_position[j] = new_velocity[j] + multi_swarm_vector[i_sub_swarm, i_particle, j]
                        end    
                    else

                        dist_temp = 0
                        dist = 0
                        for j in 1:n_particles
                            # atualiza como QSO
                            dist_temp += multi_swarm_vector[i_sub_swarm, i_particle, j] - gbest[j]^2
                            normal[j] = randn()
                            uniform = rand()
                            left_size_form[j] = rcloud * normal[j]
                        end

                        dist = real(sqrt(complex(dist_temp)))

                        if dist == 0
                            break
                        end

                        for j in 1:n_particles
                            right_size_form[j] = (uniform[j] ^ (1/dimensions_list[i_sub_swarm])) / dist
                            new_position[j] = left_size_form[j] * right_size_form[j]
                        end

                    end

                    # check if the positions is LB<x<UB
                    for i in eachindex(new_position) 
                        if new_position[i]<LB 
                            new_position[i]=LB 
                        elseif new_position[i]>UB 
                            new_position[i]=UB 
                        end
                    end 
                    
                    multi_swarm_vector[i_sub_swarm, i_particle, :] = new_position

                end   
                
                if pbest_value < gbest_value
                    gbest = copy(pbest)
                    gbest_value = copy(pbest_value)
                    context_vector[i_sub_swarm, :] = sub_swarm_pbest
                    parametros_gbest = copy(sub_swarm_pbest)

                end
            end
        
        parameters_gbest = params_reshape(parametros_gbest, params)

        A2_gbest_tv = forward_prop(X_tv, parameters_gbest)
        mse_tv = compute_cost(A2_gbest_tv, Y_tv)
        mse_treino[it_idx] = mse_tv

        A2_gbest_v = forward_prop(X_val', parameters_gbest)
        mse_v = compute_cost(A2_gbest_v, Y_val')
        mse_val[it_idx] = mse_v

        A2_gbest_t = forward_prop(X_teste', parameters_gbest)
        mse_t = compute_cost(A2_gbest_t, Y_teste')
        mse_teste[it_idx] = mse_t

        iteration += 1

        it_idx += 1
            
        end

    end
    return mse_treino, mse_val, mse_teste   
end

function n_model_cqso(X, y, n_h::Int64, num_iteracoes::Int64, perc_treino::Float64, perc_val::Float64, neutral_p::Int64, rcloud::Float64)
    
    n_in = size(X, 2)
    parametros = init_params(n_in, n_h)
    
#     best_cost = compute_cost(A2, y')
#     W1 = parametros[1]
#     b1 = parametros[2]
#     W2 = parametros[3]
#     b2 = parametros[4]

    @unpack W1, b1, W2, b2 = parametros

    dim_list = map(prod_size, parametros)
    # dim_list = [prod(size(W1)), prod(size(b1)), prod(size(W2)), prod(size(b2))]
    qtd_particulas_dim = sum(dim_list)
    # qtd_particulas_dim = convert(Int64,(size(W1)[2] + 1) * size(W1)[1] + (size(W1)[1] + 1) * size(W2)[1])

    mse_treino, mse_val, mse_teste = CQSO(X, y, parametros, qtd_particulas_dim, num_iteracoes, -1.0, 1.0, perc_treino, perc_val, neutral_p, rcloud)    
            
    return mse_treino, mse_val, mse_teste
end

# Criando cenários
function cenarios_execucoes_cqso(X, y, w, s, f::Int64, perc_treino::Float64, perc_val::Float64,qtd_execucoes::Int64, neutral_p::Int64, rcloud::Float64)
    
    # gerando os cenários dinâmicos
    X_I = cenarios_dinamicos(X, w, s)
    y_I = cenarios_dinamicos(y, w, s)
 
    # calculando a quantidade de iterações
    T = f * size(y_I)[2]

    println("Quantidade de iterações: ", T)
    
    neuronios = 2:25
    
    mse_treino = Array{Float64, 3}(undef, qtd_execucoes, size(neuronios)[1], size(y_I)[2] * f)
    mse_val = similar(mse_treino)
    mse_teste = similar(mse_treino)
 
    execucoes = 1:qtd_execucoes

    for execucao in eachindex(execucoes)

        println("Execução: ", execucao)
        
        # Neuronios
        for (j,z) in zip(neuronios, eachindex(neuronios))

            println("Neurônios: ", j)
            
            mse_treino_lista_temp, mse_val_lista_temp, mse_teste_lista_temp = n_model_cqso(X_I, y_I, j, f, perc_treino, perc_val, neutral_p, rcloud)

            # salvar lista com os mse de treino para todas as iterações
            mse_treino[execucao, z,:] = mse_treino_lista_temp
            # salvar lista com os mse de validacao para todas as iteracoes
            mse_val[execucao, z,:] = mse_val_lista_temp
            # salvar lista com os mse de teste para todas as iterações
            mse_teste[execucao, z,:] = mse_teste_lista_temp

        end
    end
    return mse_treino, mse_val, mse_teste
end

function run_model_save_output(X, y, w, s, f, experimento, algoritmo, dataset, cenario)

    quantidade_janelas = convert(Int64,floor((size(y)[1] - w)/s))

    println("# Algoritmo: $algoritmo \n ## Dataset: $experimento")
    println("### Cenario $cenario")
    println("### Tamanho da janela (w): $w")
    println("### Tamanho do passo (s): $s")
    println("### Quantidade de iteracoes (f): $f")


    if algoritmo == "pso"
        
        @time dados_mse_treino, dados_mse_val, dados_mse_teste = cenarios_execucoes_pso(X, y, w, s, f, 0.54, 0.24, 3)
        dados_resultados, dados_resultados_mse_treino, dados_resultados_mse_teste = avaliacao_resultados(dados_mse_treino, dados_mse_val, dados_mse_teste, f, quantidade_janelas, 3)

    elseif algoritmo == "cqso"
        @time dados_mse_treino, dados_mse_val, dados_mse_teste = cenarios_execucoes_cqso(X, y, w, s, f, 0.54, 0.24, 3, 25, 0.2)
        dados_resultados, dados_resultados_mse_treino, dados_resultados_mse_teste = avaliacao_resultados(dados_mse_treino, dados_mse_val, dados_mse_teste, f, quantidade_janelas, 3)

    end


    output1 = "resultados/$(dataset)_resultados_$(experimento)_$(algoritmo)_$(cenario).csv"
    output2 = "resultados/$(dataset)_resultados_mse_treino_$(experimento)_$(algoritmo)_$(cenario).csv"
    output3 = "resultados/$(dataset)_resultados_mse_teste_$(experimento)_$(algoritmo)_$(cenario).csv"
    CSV.write(output1,DataFrame(dados_resultados'))
    CSV.write(output2,DataFrame(dados_resultados_mse_treino))
    CSV.write(output3,DataFrame(dados_resultados_mse_teste))

end

function run(args)

    algoritmo = lowercase(args["algoritmo"])
    dataset = lowercase(args["dataset"])
    cenario = string(args["cenario"])

    if dataset == "sunspot"
        experimento = "sunspot"
        sunspot = CSV.read("dados/sunspot.csv", DataFrame)
        sunspot_serie = sunspot["valor"]
        sunspot_norm = normalizar(sunspot_serie)
        qtd_inputs = 10
        X, y = split_sequence(sunspot_norm, qtd_inputs);

        exp = Dict(
            "1" => [60, 10, 50],
            "2" => [60, 20, 100],
            "3" => [60, 40, 150],
            "4" => [60, 60, 100])

        if algoritmo == "pso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3], 
            experimento, algoritmo, dataset, cenario)
        elseif algoritmo == "cqso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        end

    elseif dataset == "airline"

        experimento = "airline"
        airline = CSV.read("dados/airline_passengers.csv", DataFrame)
        airline_serie = airline["valor"]
        airline_norm = normalizar(airline_serie)
        qtd_inputs = 12
        X,y = split_sequence(airline_norm, qtd_inputs);

        exp = Dict(
            "1" => [32, 5, 50],
            "2" => [32, 10, 100],
            "3" => [32, 25, 150],
            "4" => [32, 32, 100])

        if algoritmo == "pso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        elseif algoritmo == "cqso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        end

    elseif dataset == "aws"

        experimento = "aws"
        aws = CSV.read("dados/wine_sales.csv", DataFrame)
        aws_serie = aws["valor"]
        aws_norm = normalizar(aws_serie)
        qtd_inputs = 12
        X,y = split_sequence(aws_norm, qtd_inputs);

        exp = Dict(
            "1" => [42, 5, 50],
            "2" => [42, 20, 100],
            "3" => [42, 35, 150],
            "4" => [42, 32, 100])

        if algoritmo == "pso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        elseif algoritmo == "cqso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3], 
            experimento, algoritmo, dataset, cenario)
        
        end
        
    elseif dataset == "sp500"
        experimento = "sp500"
        sp500 = CSV.read("dados/sp500.csv", DataFrame)
        sp500_serie = sp500["valor"]
        sp500_norm = normalizar(sp500_serie)
        qtd_inputs = 4
        X,y = split_sequence(sp500_norm, qtd_inputs);

        exp = Dict(
            "1" => [58, 10, 50],
            "2" => [58, 20, 100],
            "3" => [58, 40, 150],
            "4" => [58, 58, 100])

        if algoritmo == "pso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        elseif algoritmo == "cqso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        
        end

    elseif dataset == "usd"
        experimento = "usd"
        usd = CSV.read("dados/usa_accident_death.csv", DataFrame)
        usd_serie = usd["valor"]
        usd_norm = normalizar(usd_serie)
        qtd_inputs = 12
        X,y = split_sequence(usd_norm, qtd_inputs);

        exp = Dict(
            "1" => [20, 2, 50],
            "2" => [20, 8, 100],
            "3" => [20, 16, 150],
            "4" => [20, 20, 100])

        if algoritmo == "pso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        elseif algoritmo == "cqso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        
        end
        
    elseif dataset == "hit"
        experimento = "hit"
        hit = CSV.read("dados/internet_traffic.csv", DataFrame)
        hit_serie = hit["valor"]
        hit_norm = normalizar(hit_serie)
        qtd_inputs = 24
        X,y = split_sequence(hit_norm, qtd_inputs);

        exp = Dict(
            "1" => [584, 100, 50],
            "2" => [584, 250, 100],
            "3" => [584, 500, 150],
            "4" => [584, 584, 50])

        if algoritmo == "pso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        elseif algoritmo == "cqso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        
        end

    elseif dataset == "dmt"
        experimento = "dmt"
        dmt = CSV.read("dados/daily_temp.csv", DataFrame)
        dmt_serie = dmt["valor"]
        dmt_norm = normalizar(dmt_serie)
        qtd_inputs = 30
        X,y = split_sequence(dmt_norm, qtd_inputs);

        exp = Dict(
            "1" => [510, 100, 50],
            "2" => [510, 200, 100],
            "3" => [510, 400, 150],
            "4" => [510, 510, 100])

        if algoritmo == "pso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        elseif algoritmo == "cqso"
            run_model_save_output(X, y, exp[cenario][1], exp[cenario][2], exp[cenario][3],
            experimento, algoritmo, dataset, cenario)
        
        end

    end

end


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin

        "--algoritmo", "-a"
            help = "Algoritmos disponíveis: PSO, CQSO"
            arg_type = String
            required = true

        "--dataset", "-d"
            help = "Datasets disponíveis para teste: sunspot, ariline, aws, sp500, usd, hit e dmt"
            arg_type = String
            required = true

        "--cenario", "-c"
            help = "Número do cenário a ser rodado, disponíveis: 1, 2, 3 e 4"
            arg_type = String
            required = true

    end

    return parse_args(s)

end

function main()
    parsed_args = parse_commandline()
    for (arg, val) in parsed_args
        println("  $arg ==>  $val")
    end

    run(parsed_args)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


