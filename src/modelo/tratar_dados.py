"""
Este modulo trata os dados do modelo.

Este modulo faz o tratamento inicial dos dados a serem
utilizados no autoencoder. É possivel filtrar os dados
inciais; é possivel selecionar um intervalo para os dados;
e levar os dados para a mesma escala.

"""

import sys

import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from configs.DataConfig import data_settings

sys.path.append('src')
sys.path.append('modelo')


def filtrar_horas_iniciais(dados) -> pd.DataFrame:
    """
    Corta a quantidade de horas iniciais para estabilização do sistema.

    Corta a quantidade de horas iniciais para estabilização
    do sistema devido aos dados do sistema serem compostos
    também por dados coletados no aquecimento da planta e
    tais dados nao refletirem a condição de operação regulares.

    Args:
        dados (DataFrame): Dados a serem cortados.

    Returns:
        DataFrame: Dados cortados
    """
    logger.info(f'Primeiras {data_settings.horas_cortadas} foram descartadas')
    return dados[data_settings.horas_cortadas:].reset_index(drop=True)


def selecionar_intervalo(dados):
    """
    Pega apenas os dados dentro de um intervalo de minutos.

    Os dados originais são dados a cada minuto, essa função
    permite retornar os dados em determinado intervalo. Por
    exemplo: se o intervalo é 5 ela retorna os dados a cada
    5 minutos.

    Args:
        dados (DataFrame): Dados a cada minuto.

    Returns:
        DataFrame: Apenas os dados no intervalo selecionado.
    """
    if data_settings.flag:
        dados = dados[dados.index % data_settings.intervalo == 0]
        logger.info(f'Selecionando dados a cada {data_settings.intervalo} min')
        return dados.reset_index(drop=True)


def scale_data(dados) -> pd.DataFrame:
    """
    Essa função leva os dados para a mesma escala.

    Os dados originais estão em escalas diferentes
    e esta função aplica o StandardScaler do pacote
    SKlearn para trazer todos os dados para a mesma
    escala.

    Args:
        dados (DataFrame): Dados a serem transformados.

    Returns:
        DataFrame: Dados rescalados.
    """
    ss = StandardScaler()
    logger.info('Trazendo os dados para a mesma escala')
    return pd.DataFrame(ss.fit_transform(dados), columns=dados.columns)
