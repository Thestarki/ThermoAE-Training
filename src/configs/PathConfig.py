"""Este modulo contém todas as configuraçoes do modelo."""

from pydantic import FilePath
from pydantic_settings import BaseSettings


class PathSettings(BaseSettings):
    """
    Uma classe de serviço para reunir todos os caminhos ajustáveis do modelo.

    Esta classe configura todo os caminhos do modelo desenvolvido,
    aqui estao reunidos todos os parametros modificaveis no projeto.

    Atributos:
        data_path: caminho para o arquivo excel com os dados de referencia
        caminho_treino: local onde sera armazenado o csv com os dados de treino
        caminho_teste: local onde sera armazenado o csv com os dados de teste
        path_to_model: caminho que o modelo treinado sera salvo
        path_to_db: caminho da base de dados

    """

    # FilePaths
    data_path: FilePath = (
        'C:/Users/fabri/Desktop/Python/Auto_encoder/Dados/' +
        'Reference_data_1min.xlsx'
        )

    caminho_treino: str = (
        'C:/Users/fabri/Desktop/Python/Auto_encoder/Dados/' +
        'Dados_gerados/treino_referencia.csv'
        )

    caminho_teste: str = (
        'C:/Users/fabri/Desktop/Python/Auto_encoder/Dados/' +
        'Dados_gerados/teste_referencia.csv'
        )

    path_to_model: str = (
        'C:/Users/fabri/Desktop/Python/Auto_encoder/Dados/' +
        'Dados_gerados/'
        )

    path_to_db: str = 'src//db//dbsqlite'


path_settings = PathSettings()
