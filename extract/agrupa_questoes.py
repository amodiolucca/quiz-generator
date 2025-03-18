import pandas as pd

def processar_csv(arquivo_entrada, arquivo_saida):
    """
    Carrega um CSV, agrupa os dados pelo campo 'CO_PROVA', ordena pelo campo 'CO_POSICAO'
    e salva o resultado em um novo arquivo CSV.
    """
    df = pd.read_csv(arquivo_entrada, sep=";", encoding='ISO-8859-1')
    df_agrupado = df.groupby('CO_PROVA').apply(lambda x: x.sort_values(by='CO_POSICAO'))
    df_agrupado = df_agrupado.reset_index(drop=True)
    df_agrupado.to_csv(arquivo_saida, index=False)

# Exemplo de uso
processar_csv('itens_prova_2016.csv', 'itensTRI.csv')
