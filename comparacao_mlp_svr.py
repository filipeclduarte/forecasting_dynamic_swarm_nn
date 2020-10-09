

import numpy as np
import pandas as pd
import pickle


def main():
    '''
    Função principal que lê os resultados das prop para os datasets
    Salva em dicionário e armazena em pickle e .csv
    os arquivos .csv estão concatenados

    '''
    datasets = ["sunspot", "airline", "aws", "sp500", "usd"]
    algoritmos = ["pso", "cqso"]
    cenarios = [1, 2, 3, 4]

    resultados = dict.fromkeys(datasets)
    for dataset in datasets:
        algoritmos_dict = dict.fromkeys(algoritmos)
        for algoritmo in algoritmos:
            cenarios_dict = dict.fromkeys(cenarios)
            for cenario in cenarios:
                # ler dados
                file = f'resultados/prop_mlp_venceu_{dataset}_resultados_{dataset}_{algoritmo}_{cenario}.csv'
                df = pd.read_csv(file)
                cenarios_dict[cenario] = df
            algoritmos_dict[algoritmo] = cenarios_dict
        resultados[dataset] = algoritmos_dict

        ## mostrando em dataframe
        lista_df_pso = [
                        resultados[dataset]["pso"][1], 
                        resultados[dataset]["pso"][2], 
                        resultados[dataset]["pso"][3], 
                        resultados[dataset]["pso"][4],
                        ]

        lista_df_cqso = [
                        resultados[dataset]["cqso"][1], 
                        resultados[dataset]["cqso"][2], 
                        resultados[dataset]["cqso"][3], 
                        resultados[dataset]["cqso"][4],
                        ]

        # Atribuindo os nomes dos cenários
        df_pso = pd.concat(lista_df_pso, axis = 1)
        df_pso.columns = ['1', '2', '3', '4']
        df_cqso = pd.concat(lista_df_cqso, axis = 1)
        df_cqso.columns = ['1', '2', '3', '4']
        
        # salvando em .csv
        df_pso.to_csv(f'resultados/df_resultados_comparativos_{dataset}_pso.csv')
        df_cqso.to_csv(f'resultados/df_resultados_comparativos_{dataset}_cqso.csv')

    # Salvando em pickle
    with open('comparacao_mlp_svr.pickle', 'wb') as handle:
        pickle.dump(resultados, handle)
    

if __name__ == "__main__":
    main()