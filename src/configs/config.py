"""Este modulo contém todas as configuraçoes do modelo."""

import os
from typing import List

import torch
from loguru import logger
from pydantic import FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine

DOTENV = os.path.join(os.path.dirname(__file__), '.env')


class Settings(BaseSettings):
    """
    Uma classe para reunir todos os parametros ajustáveis do autoencoder.

    Esta classe configura todo o modelo desenvolvido, aqui estao
    reunidos todos os parametros modificaveis no projeto.

    Atributos:

        batch_size: tamanho de batch utilizado no treinamento
        hidden_size: tamanho da camada oculta da rede
        epocas: numero de vezes que os dados passarao na rede

    """

    model_config = SettingsConfigDict(
        env_file=DOTENV,
        env_file_encoding='utf-8',
    )

    # Neural Network configuration
    batch_size: int

    hidden_size: int

    epochs: int

    variables: List

    horas_cortadas: int

    flag: bool
    intervalo: int

    perc_treino: float
    perc_teste: float

    # FilePaths
    data_path: str

    caminho_treino: str

    caminho_teste: str

    path_to_model: str

    path_to_db: FilePath


def checar_gpu() -> torch.device:
    """Funcao para verificar se tem gpu disponivel.

    Returns:
        _type_: Retorna o dispositivo usado
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def loggerr():
    """Criando o Logger."""
    #logger.remove()
    logger.add(
        'info.log',
        rotation='1 MB',
        compression='zip',
        retention='1 MINUTE',
        level='DEBUG',
    )


settings = Settings()

engine = create_engine(os.path.join('sqlite:///', settings.path_to_db))
