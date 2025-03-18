import pandas as pd

def carregar_planilha(caminho):
    """Carrega um arquivo CSV ou Excel com base na extensão."""
    if caminho.endswith('.csv'):
        return pd.read_csv(caminho, sep="," if "csv" in caminho else ";", encoding="ISO-8859-1")
    else:
        return pd.read_excel(caminho)

def atualizar_planilha(planilha1_path, planilha2_path, co_prova, output_path):
    """
    Atualiza a planilha1 com base nos dados da planilha2, filtrando pelo código da prova (CO_PROVA). O código da prova e a posição da questão na prova
    são as chaves em comum entre as planilhas, e serão usados para fazer a junção delas, uma espécie de join das tabelas a partir do código da prova e
    posição da questão.
    """
    df1 = carregar_planilha(planilha1_path)
    df2 = carregar_planilha(planilha2_path)

    # Cria as colunas apenas se não existirem
    for col in ['Gabarito', 'parA', 'parB', 'parC']:
        if col not in df1.columns:
            df1[col] = ''

    # Percorre a planilha 1 e atualiza os campos
    for i, row in df1.iterrows():
        filtro = (df2['CO_POSICAO'] == row['Numero']) & (df2['CO_PROVA'] == co_prova) #busca o número da questão pelo CO_POSICAO e o número da prova pelo CO_PROVA
        correspondente = df2[filtro]

        if not correspondente.empty:
            df1.at[i, 'Gabarito'] = correspondente['TX_GABARITO'].values[0]
            df1.at[i, 'parA'] = correspondente['NU_PARAM_A'].values[0]
            df1.at[i, 'parB'] = correspondente['NU_PARAM_B'].values[0]
            df1.at[i, 'parC'] = correspondente['NU_PARAM_C'].values[0]
        else:
            print(f"Sem correspondência para {row['Numero']}, mantendo valores originais.")

    # Salvar a nova planilha no formato correto
    if output_path.endswith('.csv'):
        df1.to_csv(output_path, index=False)
    else:
        df1.to_excel(output_path, index=False)

atualizar_planilha('p2_dia1.csv', 'arquivo_agrupado_ordenado.csv', 291, 'planilha_atualizada.csv')
