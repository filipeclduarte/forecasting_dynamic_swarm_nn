using Distributions
using CSV
using Plots
using DataFrames
using Random
using ArgParse
using ScikitLearn

@sk_import svm: SVR 

Random.seed!(123);

# normalizar serie
function normalizar(serie)
    max = maximum(serie)
    min = minimum(serie)
    
    y_temp = 2 .* ((serie .- min)) ./ (max .- min) .- 1
    
    y = y_temp ./ sqrt(size(serie)[1])
    
    return y
end

function desnormalizar(serie_norm, serie)
    max = maximum(serie)
    min = maximum(serie)
    
    serie_temp = serie_norm .* sqrt(size(serie)[1])
    serie_temp2 = (serie_temp .+ 1)/2
    serie_temp3 = serie_temp2 * ((max - min) + min)
    return serie_temp3 
end

function split_sequence(serie, n_steps_in::Int64)
    len = size(serie)[1]
    max_iter = len - n_steps_in
    seq_x = zeros(max_iter, n_steps_in)
    seq_y = zeros(max_iter)
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

function compute_cost(pred, y)
    m = size(y)[1]
    erro = pred .- y
    cost = 1/m * sum(erro.^2)
    return cost
end

# PSO function
function PSO(X, y, n_particles::Int64, max_iter::Int64, LB, UB, perc_treino::Float64, perc_val::Float64)
    
    mse_treino = zeros(size(y)[2]*max_iter)
    mse_val = zeros(size(y)[2]*max_iter)
    mse_teste = zeros(size(y)[2]*max_iter)
    
    dim = 3
    
    particles = zeros(n_particles, dim)
    particles[:, 1] = rand(Uniform(LB[1], UB[1]), n_particles)
    particles[:, 2] = rand(Uniform(LB[2], UB[2]), n_particles)
    particles[:, 3] = rand(Uniform(LB[3], UB[3]), n_particles)
    
    velocity = zeros(n_particles, dim)
    
    modelo = fit!(SVR(kernel = "rbf", C = particles[1,1], epsilon = particles[1, 2], gamma = particles[1,3]), X[:,:,1], y[:, 1])
    y_pred = predict(modelo, X[:,:,1])
    gbest_value = compute_cost(y_pred, y[:,1])    

    fitness_value = zeros(n_particles)

    for i in eachindex(fitness_value)
        modelo = fit!(SVR(kernel = "rbf", C = particles[i,1], epsilon = particles[i, 2], gamma = particles[i,3]), X[:,:,1], y[:, 1])
        y_pred = predict(modelo, X[:, :, 1])
        fitness_value[i] = compute_cost(y_pred, y[:,1])
    end
    
    id_min = argmin(fitness_value)
    pbest = copy(particles)
    gbest = copy(pbest[id_min, :])

    wmax = 0.9
    wmin = 0.4
    c1 = 1.5
    c2 = 1.5
    
    iteracao = 1
    for janela in 1:size(y)[2]
        
        X_treino, Y_treino, X_teste, Y_teste, X_val, Y_val = divisao_dados_temporais(X[:,:,janela], y[:,janela], perc_treino, perc_val)
        
        X_tv = vcat(X_treino, X_val)
        Y_tv = vcat(Y_treino, Y_val)
        
        for k in 1:max_iter
            w=wmax-(wmax-wmin)*k/max_iter
        
            for i in 1:n_particles
                velocity[i, :] = w*velocity[i,:] .+ c1*rand().*(pbest[i,:] .- particles[i,:]) .+ c2*rand().*(gbest[:] .- particles[i,:])
                particles[i,:] = particles[i,:] .+ velocity[i,:]
            end
            
            # handling boundary violations 
            for i in 1:n_particles
                for j in 1:dim
                    if particles[i,j]<LB[j] 
                        particles[i,j]=LB[j] 
                    elseif particles[i,j]>UB[j] 
                        particles[i,j]=UB[j] 
                    end
                end
            end 

            # evaluating fitness
            for i in 1:n_particles
                modelo = fit!(SVR(kernel = "rbf", C = particles[i,1], epsilon = particles[i, 2], gamma = particles[i,3]), X_treino, Y_treino)
                y_pred = predict(modelo, X_treino)
                fitness_value[i] = compute_cost(y_pred, Y_treino)
                
                # pbest
                modelo_pbest = fit!(SVR(kernel = "rbf", C = pbest[i,1], epsilon = pbest[i, 2], gamma = pbest[i,3]), X_treino, Y_treino)
                y_pred_pbest = predict(modelo, X_treino)

                if fitness_value[i] < compute_cost(y_pred_pbest, Y_treino)
                    pbest[i, :] = copy(particles[i, :])
                end
                
                # gbest
                if fitness_value[i] < gbest_value
                    gbest_value = fitness_value[i]
                    gbest = copy(particles[i,:])   
                end

            end

            # updating pbest and fitness
            # for i in 1:n_particles
            #     modelo_pbest = fit!(SVR(kernel = "rbf", C = pbest[i,1], epsilon = pbest[i, 2], gamma = pbest[i,3]), X_treino, Y_treino)
            #     y_pred_pbest = predict(modelo, X_treino)
            #     if fitness_value[i] < compute_cost(y_pred_pbest, Y_treino)
            #         pbest[i, :] = copy(particles[i, :])
            #     end
            # end

            # updating gbest 
            # for i in 1:n_particles    
            #     if fitness_value[i] < gbest_value
            #         gbest_value = fitness_value[i]
            #         gbest = copy(particles[i,:])   
            #     end
            # end
    
            # treinamento mse
            modelo_iter = fit!(SVR(kernel = "rbf", C = gbest[1], epsilon = gbest[2], gamma = gbest[3]), X_tv, Y_tv)
            pred_gbest_tv = predict(modelo_iter, X_tv)
            mse_tv = compute_cost(pred_gbest_tv, Y_tv)
            mse_treino[iteracao] = mse_tv

            # validacao mse
            pred_gbest_val = predict(modelo_iter, X_val)
            mse_v = compute_cost(pred_gbest_val, Y_val)
            mse_val[iteracao] = mse_v

            # teste
            pred_gbest_t = predict(modelo_iter, X_teste)
            mse_t = compute_cost(pred_gbest_t, Y_teste)
            mse_teste[iteracao] = mse_t
            
            iteracao += 1
        end
        
    end
    return mse_treino, mse_val, mse_teste
end

function init_params()
   
    C = rand(Uniform(10, 1000))
    epsilon = rand(Uniform(1e-5, 1e-3))
    gamma = rand(Uniform(1e-2, 1))
    
    params = Dict("C" => C, "epsilon" => epsilon, "gamma" => gamma)
    
    return params
    
end

function svr_model_pso(X, y, num_iteracoes::Int64, perc_treino::Float64, perc_val::Float64)

    
    LB = [1, 1e-5, 1e-2]
    UB = [1000, 1e-3, 1]
    
    mse_treino, mse_val, mse_teste = PSO(X, y, 50, num_iteracoes, LB, UB, perc_treino, perc_val)
        
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
    end
    return cenarios
end

# Criando cenários
function cenarios_execucoes_pso(X, y, w, s, f::Int64, perc_treino::Float64, perc_val::Float64,qtd_execucoes::Int64)
    
    # gerando os cenários dinâmicos
    X_I = cenarios_dinamicos(X, w, s)
    y_I = cenarios_dinamicos(y, w, s)
 
    # calculando a quantidade de iterações
    T = f * size(y_I)[2] 

    println("Quantidade de iterações: ", T)
    
    
    mse_treino = zeros(qtd_execucoes, size(y_I)[2] * f)
    mse_val = zeros(qtd_execucoes, size(y_I)[2] * f)
    mse_teste = zeros(qtd_execucoes, size(y_I)[2] * f)
 
    execucoes = 1:qtd_execucoes

    for execucao in eachindex(execucoes)

        println("Execução: ", execucao)
            
        mse_treino_lista_temp, mse_val_lista_temp, mse_teste_lista_temp = svr_model_pso(X_I, y_I, f, perc_treino, perc_val)

        # salvar lista com os mse de treino para todas as iterações
        mse_treino[execucao, :] = mse_treino_lista_temp
        # salvar lista com os mse de validacao para todas as iteracoes
        mse_val[execucao, :] = mse_val_lista_temp
        # salvar lista com os mse de teste para todas as iterações
        mse_teste[execucao, :] = mse_teste_lista_temp

    end
    
    return mse_treino, mse_val, mse_teste
    
end
      
function avaliacao_resultados(mse_treino_cenarios, mse_val_cenarios, mse_teste_cenarios, f, quantidade_janelas, execucoes)
            
    qtd_iteracoes = size(mse_treino_cenarios)[2]
    
    te = sum(mse_treino_cenarios, dims = 1)./qtd_iteracoes
    
    ge = sum(mse_teste_cenarios, dims = 1)./qtd_iteracoes

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

    return resultados, mse_treino_cenarios, mse_teste_cenarios

end  

# CQSO
function CQSO(X, y, n_particles::Int64, max_iter::Int64, LB, UB, perc_treino::Float64, perc_val::Float64, neutral_p::Int64, rcloud::Float64)
    
    mse_treino = zeros(size(y)[2]*max_iter)
    mse_val = zeros(size(y)[2]*max_iter)
    mse_teste = zeros(size(y)[2]*max_iter)
    
    # quantidade de colunas
    dim = 3

    n = n_particles
    a = zeros(Int64, n)

    for i in 2:n
        if n%i == 0
            a[i] = i
        end
    end

    n_sub_swarms = sort(a[a.!=0])[1]

    # divide the dimensions per subswarm
    num = n
    div = n_sub_swarms

    dimensions_list = zeros(Int64, div)

    for x in 1:div
        if x < num % div
            dimensions_list[x] = 1 + (num ÷ div)
        else
            dimensions_list[x] = 0 + (num ÷ div)
        end
    end

    context_vector = zeros(n_sub_swarms, dim)

    # Create a multiswarm and his velocities
    multi_swarm_vector = zeros(n_sub_swarms, n, dim)
    velocity_vector = zeros(n_sub_swarms, n, dim)
    
    for i_subswarm in 1:n_sub_swarms  
        for i_particle in 1:n_particles
            for j_particle in 1:dim
                context_vector[i_subswarm, j_particle] = rand(Uniform(LB[j_particle], UB[j_particle]))
                multi_swarm_vector[i_subswarm, i_particle, j_particle] = rand(Uniform(LB[j_particle], UB[j_particle]))
            end
        end
    end

    gbest = copy(multi_swarm_vector[1,1,:])
    pbest = copy(multi_swarm_vector[1,1,:])

    sub_swarm_pbest = copy(context_vector[1,:])

    modelo = fit!(SVR(kernel = "rbf", C = sub_swarm_pbest[1], epsilon = sub_swarm_pbest[2], gamma = sub_swarm_pbest[3]), X[:,:,1], y[:, 1])
    
    y_pred = predict(modelo, X[:, :, 1])
    gbest_value = compute_cost(y_pred, y[:, 1])
    pbest_value = copy(gbest_value)

    parametros_gbest = copy(sub_swarm_pbest)
        
    wmax = 0.9
    wmin = 0.4
    c1 = 1.5
    c2 = 1.5
    
    it_idx = 1

    for janela in 1:size(y)[2]

        X_treino, Y_treino, X_teste, Y_teste, X_val, Y_val = divisao_dados_temporais(X[:,:, janela], y[:, janela], perc_treino, perc_val)

        X_tv = vcat(X_treino, X_val)
        Y_tv = vcat(Y_treino, Y_val)

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
                    particle = copy(multi_swarm_vector[i_sub_swarm,i_particle,:])
                    context_copy[i_sub_swarm, :] = copy(particle)

                    modelo = fit!(SVR(kernel = "rbf", C = context_copy[i_sub_swarm,1], epsilon = context_copy[i_sub_swarm, 2], 
                                gamma = context_copy[i_sub_swarm,3]), X_treino, Y_treino)
                    
                    y_pred = predict(modelo, X_treino)
                    fitness_candidate = compute_cost(y_pred, Y_treino)
                    
                    if fitness_candidate < pbest_value
                    # se o fitness da nova particula for melhor, ela vira o pbest
                        pbest = copy(multi_swarm_vector[i_sub_swarm, i_particle, :])
                        pbest_value = copy(fitness_candidate)
                        sub_swarm_pbest = copy(context_copy[i_sub_swarm, :])
                         # feito o pbest devemos atualizar as posições das particulas
                    end

                    if i_particle <= (neutral_p - 1)  

                        # atualiza como PSO vanilla
                        new_velocity = (w*velocity_vector[i_sub_swarm, i_particle, :]) .+ 
                        ((c1 * rand()) * (pbest .- multi_swarm_vector[i_sub_swarm, i_particle, :])) .+
                        ((c2 * rand()) * (gbest .- multi_swarm_vector[i_sub_swarm, i_particle, :]))

                        new_position = new_velocity .+ multi_swarm_vector[i_sub_swarm, i_particle, :]
                        
                        
                    else
                        # atualiza como QSO
                        dist = sqrt(sum((multi_swarm_vector[i_sub_swarm, i_particle, :] .- gbest).^2))
                        normal = rand(Normal(0, 1), dim)
                        uniform = rand(dim)
                        left_size_form = rcloud .* normal
                        
                        
                        if dist == 0
                            break

                        end

                        right_size_form = (uniform.^(1/dimensions_list[i_sub_swarm])) ./ dist
                        new_position = left_size_form .* right_size_form
                            
                        
                    end
                        
                    # handling boundary violations 
                    for j in 1:dim
                        if new_position[j]<LB[j] 
                            new_position[j]=LB[j] 
                        elseif new_position[j]>UB[j] 
                            new_position[j]=UB[j] 
                        end
                    end 

                    
                    multi_swarm_vector[i_sub_swarm, i_particle, :] = copy(new_position)

                end   

                if pbest_value < gbest_value
                    gbest = copy(pbest)
                    gbest_value = copy(pbest_value)
                    context_vector[i_sub_swarm, :] = copy(sub_swarm_pbest)
                    parametros_gbest = copy(sub_swarm_pbest)
                end
            
                
            end

            # treinamento mse
            modelo_iter = fit!(SVR(kernel = "rbf", C = gbest[1], epsilon = gbest[2], gamma = gbest[3]), X_tv, Y_tv)

            pred_gbest_tv = predict(modelo_iter, X_tv)
            mse_tv = compute_cost(pred_gbest_tv, Y_tv)
            mse_treino[it_idx] = mse_tv
            
#             println("MSE treinamento e validação: $mse_tv")

            # validacao mse
            pred_gbest_val = predict(modelo_iter, X_val)
            mse_v = compute_cost(pred_gbest_val, Y_val)
            mse_val[it_idx] = mse_v
            
#             println("MSE validação:               $mse_v")

            # teste
            pred_gbest_t = predict(modelo_iter, X_teste)
            mse_t = compute_cost(pred_gbest_t, Y_teste)
            mse_teste[it_idx] = mse_t
            
#             println("MSE teste:                   $mse_tv")

            it_idx += 1
            iteration += 1
        end
    end
                         
    return mse_treino, mse_val, mse_teste
end
    
    
function svr_model_cqso(X, y, num_iteracoes::Int64, perc_treino::Float64, perc_val::Float64, neutral_p::Int64, rcloud::Float64)

    parametros = init_params()
    
    qtd_particulas_dim = length(parametros)    
    
    LB = [1, 1e-5, 1e-2]
    UB = [1000, 1e-3, 1]
    
    mse_treino, mse_val, mse_teste = CQSO(X, y, 50, num_iteracoes, LB, UB, perc_treino, perc_val, neutral_p, rcloud)
        
    return mse_treino, mse_val, mse_teste
end


# Criando cenários
function cenarios_execucoes_cqso(X, y, w, s, f::Int64, perc_treino::Float64, perc_val::Float64, qtd_execucoes::Int64, neutral_p::Int64, rcloud::Float64)
    
    # gerando os cenários dinâmicos
    X_I = cenarios_dinamicos(X, w, s)
    y_I = cenarios_dinamicos(y, w, s)
 
    # calculando a quantidade de iterações
    T = f * size(y_I)[2] 

    println("Quantidade de iterações: ", T)
    
    
    mse_treino = zeros(qtd_execucoes, size(y_I)[2] * f)
    mse_val = zeros(qtd_execucoes, size(y_I)[2] * f)
    mse_teste = zeros(qtd_execucoes, size(y_I)[2] * f)
 
    execucoes = 1:qtd_execucoes

    for execucao in eachindex(execucoes)

        println("Execução: ", execucao)
            
        mse_treino_lista_temp, mse_val_lista_temp, mse_teste_lista_temp = svr_model_cqso(X_I, y_I, f, perc_treino, perc_val, neutral_p, rcloud)

        # salvar lista com os mse de treino para todas as iterações
        mse_treino[execucao, :] = mse_treino_lista_temp
        # salvar lista com os mse de validacao para todas as iteracoes
        mse_val[execucao, :] = mse_val_lista_temp
        # salvar lista com os mse de teste para todas as iterações
        mse_teste[execucao, :] = mse_teste_lista_temp

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

    output1 = "resultados/svr_$(dataset)_resultados_$(experimento)_$(algoritmo)_$(cenario).csv"
    output2 = "resultados/svr_$(dataset)_resultados_mse_treino_$(experimento)_$(algoritmo)_$(cenario).csv"
    output3 = "resultados/svr_$(dataset)_resultados_mse_teste_$(experimento)_$(algoritmo)_$(cenario).csv"
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


