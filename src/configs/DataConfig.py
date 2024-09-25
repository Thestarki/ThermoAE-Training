"""Este modulo contém todas as configuraçoes do modelo."""

from pydantic_settings import BaseSettings
from sqlalchemy import create_engine

from configs.PathConfig import path_settings


class DataSettings(BaseSettings):
    """
    Uma classe de serviço para reunir todos os parametros ajustáveis do modelo.

    Esta classe configura todo o modelo desenvolvido, aqui estao
    reunidos todos os parametros modificaveis no projeto.

    Atributos:
        data_path: caminho para o arquivo excel com os dados de referencia
        caminho_treino: local onde sera armazenado o csv com os dados de treino
        caminho_teste: local onde sera armazenado o csv com os dados de teste
        path_to_model: caminho que o modelo treinado sera salvo
        path_to_db: caminho da base de dados

        variables: lista das caracteristicas que serao utilizadas no
        treinamento horas_cortada: quantidade de horas iniciais de operacao
        que serao ignoradas flag: indica se há a necessidade de selecionar os
        dados em um intervalo intervalo: janela de tempo extraida dos dados
        originais
    """

    # Data aquiring and transformig
    variables: list = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']

    horas_cortadas: int = 6000

    flag: bool = False
    intervalo: int = 5

    perc_treino: float = 0.7
    perc_teste: float = 0.3


data_settings = DataSettings()

# SQLalchemy data engine initialization
engine = create_engine('sqlite:///' + path_settings.path_to_db)
