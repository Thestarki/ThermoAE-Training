"""Este modulo contém todas as configuraçoes do modelo."""

from pydantic_settings import BaseSettings

from configs.DataConfig import data_settings


class SettingsAE(BaseSettings):
    """
    Uma classe para reunir todos os parametros ajustáveis do autoencoder.

    Esta classe configura todo o modelo desenvolvido, aqui estao
    reunidos todos os parametros modificaveis no projeto.

    Atributos:

        batch_size: tamanho de batch utilizado no treinamento
        input_size: quantidade de neuronios na camada de entrada da rede
        hidden_size: tamanho da camada oculta da rede
        out_size: tamanho da camada de saida da rede
        epocas: numero de vezes que os dados passarao na rede

    """

    # Neural Network configuration
    batch_size: int = 1024

    input_size: int = len(data_settings.variables)

    hidden_size: int = 10

    out_size: int = len(data_settings.variables)

    epochs: int = 25

    # Debbug
    log_level: str = 'DEBUG'


ae_settings = SettingsAE()
