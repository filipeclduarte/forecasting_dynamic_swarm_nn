
# Avaliação dos resultados
using CSV
using DataFrames
using ArgParse

function avaliar_modelos_teste(mlp_teste, svr_teste, f, execucoes)

    mlp = zeros(execucoes, size(mlp_teste)[2])

    # quantidade_janelas = convert(Int64,round(size(mlp_teste)[2]/f, digits = 0)) - 1

    for ex in 1:execucoes
        for j in 1:size(mlp)[2]
        # for janela in 1:quantidade_janelas
            if svr_teste[ex, j] > mlp_teste[ex, j]
                mlp[ex, j] = 1
            end
        end
    end
    return mlp
end

function run_analise_save_output(mlp, svr, w, s, f, experimento, algoritmo, dataset, cenario)

    println("# Algoritmo: $algoritmo \n ## Dataset: $experimento")
    println("### Cenario $cenario")
    println("### Tamanho da janela (w): $w")
    println("### Tamanho do passo (s): $s")
    println("### Quantidade de iteracoes (f): $f")

    dados_resultados = avaliar_modelos_teste(mlp, svr, f, 3)

    # dados_resultados_df = DataFrame(dados_resultados)

    # % de vezes em que a MLP venceu a SVR
    venceu = (sum(dados_resultados, dims = 2)/size(dados_resultados)[2]).*100

    for i in eachindex(venceu)
        print("MLP venceu a SVR em $(venceu[i])% \n")

    end
    # dados_resultados_df = DataFrame(hcat(mlp_resultados, svr_resultados))

    output1 = "resultados/analise_resultados_$(dataset)_resultados_$(experimento)_$(algoritmo)_$(cenario).csv"
    output2 = "resultados/prop_mlp_venceu_$(dataset)_resultados_$(experimento)_$(algoritmo)_$(cenario).csv"
    CSV.write(output1, DataFrame(dados_resultados))
    CSV.write(output2, DataFrame(venceu))
end

function run(args)

    algoritmo = lowercase(args["algoritmo"])
    dataset = lowercase(args["dataset"])
    cenario = string(args["cenario"])

    if dataset == "sunspot"
        experimento = "sunspot"
        exp = Dict(
            "1" => [60, 10, 50],
            "2" => [60, 20, 100],
            "3" => [60, 40, 150],
            "4" => [60, 60, 100])

        mlp = CSV.read("resultados/sunspot_resultados_mse_teste_sunspot_$(algoritmo)_$(cenario).csv")
        
        svr = CSV.read("resultados/svr_sunspot_resultados_mse_teste_sunspot_$(algoritmo)_$(cenario).csv", DataFrame)
        
        mlp = convert(Matrix{Float64},mlp[:, 2:end])
        svr = convert(Matrix{Float64}, svr)

        run_analise_save_output(mlp, svr, exp[cenario][1], exp[cenario][2], exp[cenario][3], 
        experimento, algoritmo, dataset, cenario)

    elseif dataset == "airline"

        experimento = "airline"
        
        exp = Dict(
            "1" => [32, 5, 50],
            "2" => [32, 10, 100],
            "3" => [32, 25, 150],
            "4" => [32, 32, 100])

        mlp = CSV.read("resultados/airline_resultados_mse_teste_airline_$(algoritmo)_$(cenario).csv", DataFrame)
        svr = CSV.read("resultados/svr_airline_resultados_mse_teste_airline_$(algoritmo)_$(cenario).csv", DataFrame)
        
        if algoritmo == "pso"
            mlp = convert(Matrix{Float64},mlp[2:end])
            svr = convert(Matrix{Float64}, svr)

        else
            mlp = convert(Matrix{Float64},mlp)
            svr = convert(Matrix{Float64}, svr)

        end

        run_analise_save_output(mlp, svr, exp[cenario][1], exp[cenario][2], exp[cenario][3],
        experimento, algoritmo, dataset, cenario)
    

    elseif dataset == "aws"

        experimento = "aws"
        
        exp = Dict(
            "1" => [42, 5, 50],
            "2" => [42, 20, 100],
            "3" => [42, 35, 150],
            "4" => [42, 32, 100])

        mlp = CSV.read("resultados/aws_resultados_mse_teste_aws_$(algoritmo)_$(cenario).csv", DataFrame)
        svr = CSV.read("resultados/svr_aws_resultados_mse_teste_aws_$(algoritmo)_$(cenario).csv", DataFrame)
        if algoritmo == "pso"
            mlp = convert(Matrix{Float64},mlp[2:end])
            svr = convert(Matrix{Float64}, svr)

        else
            mlp = convert(Matrix{Float64},mlp)
            svr = convert(Matrix{Float64}, svr)

        end
        run_analise_save_output(mlp, svr, exp[cenario][1], exp[cenario][2], exp[cenario][3],
        experimento, algoritmo, dataset, cenario)
        
        
    elseif dataset == "sp500"
        experimento = "sp500"
        
        exp = Dict(
            "1" => [58, 10, 50],
            "2" => [58, 20, 100],
            "3" => [58, 40, 150],
            "4" => [58, 58, 100])

        mlp = CSV.read("resultados/sp500_resultados_mse_teste_sp500_$(algoritmo)_$(cenario).csv", DataFrame)
        svr = CSV.read("resultados/svr_sp500_resultados_mse_teste_sp500_$(algoritmo)_$(cenario).csv", DataFrame)
        if algoritmo == "pso"
            mlp = convert(Matrix{Float64},mlp[2:end])
            svr = convert(Matrix{Float64}, svr)

        else
            mlp = convert(Matrix{Float64},mlp)
            svr = convert(Matrix{Float64}, svr)

        end

        run_analise_save_output(mlp, svr, exp[cenario][1], exp[cenario][2], exp[cenario][3],
        experimento, algoritmo, dataset, cenario)
    

    elseif dataset == "usd"
        experimento = "usd"
        
        exp = Dict(
            "1" => [20, 2, 50],
            "2" => [20, 8, 100],
            "3" => [20, 16, 150],
            "4" => [20, 20, 100])

        mlp = CSV.read("resultados/usd_resultados_mse_teste_usd_$(algoritmo)_$(cenario).csv", DataFrame)
        svr = CSV.read("resultados/svr_usd_resultados_mse_teste_usd_$(algoritmo)_$(cenario).csv", DataFrame)
        if algoritmo == "pso"
            mlp = convert(Matrix{Float64},mlp[2:end])
            svr = convert(Matrix{Float64}, svr)

        else
            mlp = convert(Matrix{Float64},mlp)
            svr = convert(Matrix{Float64}, svr)

        end

        run_analise_save_output(mlp, svr, exp[cenario][1], exp[cenario][2], exp[cenario][3],
        experimento, algoritmo, dataset, cenario)
    
    
    elseif dataset == "hit"
        experimento = "hit"
        
        exp = Dict(
            "1" => [584, 100, 50],
            "2" => [584, 250, 100],
            "3" => [584, 500, 150],
            "4" => [584, 584, 50])

        mlp = CSV.read("resultados/hit_resultados_mse_teste_hit_$(algoritmo)_$(cenario).csv", DataFrame)
        svr = CSV.read("resultados/svr_hit_resultados_mse_teste_hit_$(algoritmo)_$(cenario).csv", DataFrame)
        if algoritmo == "pso"
            mlp = convert(Matrix{Float64},mlp[2:end])
            svr = convert(Matrix{Float64}, svr)

        else
            mlp = convert(Matrix{Float64},mlp)
            svr = convert(Matrix{Float64}, svr)

        end
        run_analise_save_output(mlp, svr, exp[cenario][1], exp[cenario][2], exp[cenario][3],
        experimento, algoritmo, dataset, cenario)
    

    elseif dataset == "dmt"
        experimento = "dmt"
        
        exp = Dict(
            "1" => [510, 100, 50],
            "2" => [510, 200, 100],
            "3" => [510, 400, 150],
            "4" => [510, 510, 100])

        mlp = CSV.read("resultados/dmt_resultados_mse_teste_dmt_$(algoritmo)_$(cenario).csv", DataFrame)
        svr = CSV.read("resultados/svr_dmt_resultados_mse_teste_dmt_$(algoritmo)_$(cenario).csv", DataFrame)
        if algoritmo == "pso"
            mlp = convert(Matrix{Float64},mlp[2:end])
            svr = convert(Matrix{Float64}, svr)

        else
            mlp = convert(Matrix{Float64},mlp)
            svr = convert(Matrix{Float64}, svr)

        end

        run_analise_save_output(mlp, svr, exp[cenario][1], exp[cenario][2], exp[cenario][3],
        experimento, algoritmo, dataset, cenario)
    
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
