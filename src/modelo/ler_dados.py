"""
Este módulo carrega os dados de treinamento do Autoencoder.

Ele contém duas funções que possibilitam carregar os dados
de duas formas diferentes.
"""

import sys

import pandas as pd
from loguru import logger
from sqlalchemy import select

from configs.config import engine
from db.db_model import ReferenceData

sys.path.append('src')


def ler_df(data_path: str, variables: list) -> pd.DataFrame:
    """Le os dados de um excel.

    Esta função retorna os dados de referência a partir de um
    arquivo excel existente nas dependencias do programa e
    registra no logger se foi executada.

    Args:
        data_path (str): O caminho no qual os dados do excel estão localizados.
        variables (list): lista de variaveis que o modelo usa para treinamento.


    Returns:
        pd.DataFrame: Um dataframe contendo todo o conjunto de dados
    """
    logger.info(f'Carregando os dados em {data_path}')
    return pd.read_excel(
        data_path,
        usecols=variables,
    )


def ler_dados_da_db() -> pd.DataFrame:
    """
    Le os dados da base de dados.

    Esta função retorna os dados de referência a partir de uma
    base de dados do SQLite e registra no logger se foi executada.
    A leitura dos dados é feita via sqlalchemy selecinando um schema
    e uma engine.

    Returns:
        pd.DataFrame: Um dataframe contendo todo o conjunto de dados
    """
    logger.info('Carregando os dados de referencia da base de dados')
    query = select(ReferenceData)
    return pd.read_sql(query, engine)
