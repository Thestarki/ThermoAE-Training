"""
Modulo para criar dataloaders.

Este modulo cria os dataloaders que colocam os dados no
formato aceitavel pelos modulos de redes neurais do pytorch.
"""

import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from configs.config import settings

sys.path.append('src')


class ReturnTensor(Dataset):
    """
    Le os dados de um arquivo csv e transforma para tensor.

    Esta classe lê os dados de um arquivo csv e retorna os
    dados a partir de um arquivo csv existente nas dependencias
    do programa e retorno os dados como tensores. Os metodos
    descritos aqui sobrescrevem os metodos da classe Dataset.

    Argumentos:
        Dataset: Uma classe abstrata do pytorch representando
        um dataset. Essa classe permite que seja sobreescrevido
        alguns de seus metodos interno, como o __init__,
        __getitem__ e __len__ por classes definidas externamente,
        como essa classe ReturnTensor

    Metodos:
        __init__: Construtor inicial da classe, há a leitura.
        dos dados.
        __getitem__: funcao para retornar os itens como tensores.
        __len__: retorna o tamanho do dataframe.

    Retorna:
        Dataframe: Um dataframe contendo todo o conjunto de dados
    """

    def __init__(self, csv_path: str) -> None:
        """
        Funcao para ler os dados.

        Essa funcao le os dados e os conver para numpy.

        Args:
            csv_path (str): Caminho para o arquivo csv dos dados
        """
        self.dados = pd.read_csv(csv_path).to_numpy()

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Essa função seleciona as colunas dos dados.

        Args:
            idx (int): um indice a ser percorrido

        Returns:
            torch.Tensor: _description_
        """
        sample = self.dados[idx][0:len(settings.variables)]

        return torch.from_numpy(sample.astype(np.float32))

    def __len__(self) -> int:
        """Retorna o tamanho do Dataframe.

        Returns:
            int: Tamanho do dataframe
        """
        return len(self.dados)


def partir_treino_teste(dados) -> None:
    """
    Divide os dados entre treino e testes.

    Esta função embaralha os indices do dataframe original,
    separa os dados em 70% para treino e 30% para testes e
    os escreve em um arquivo csv para treino e outro para teste.

    Args:
        dados (Dataframe): Um Dataframe contendo todos os dados
    """
    indices = torch.randperm(len(dados)).tolist()

    # Getting 70% of our data to train our model
    train_size = int(settings.perc_treino * len(dados))
    df_train = dados.iloc[indices[:train_size]]

    # Getting 30% of our data to test our model
    test_size = int(settings.perc_teste * len(dados))
    df_test = dados.iloc[indices[test_size:]]

    logger.info('Escrevendo os Dataframes de treino e teste')
    # Storing the information on a file
    df_train.to_csv(settings.caminho_treino, index=False)
    df_test.to_csv(settings.caminho_teste, index=False)
    logger.info('Dataframes de treino e teste escritos!')


def chamando_return_tensor() -> torch.Tensor:
    """
    Chama a classe que transformara os dados em tensores.

    Esta função chama a classe que transforma os dados em
    tensores e retorna dois dataframes com os dados em
    tensores.

    Returns:
        torch.Tensor: Dois dataframes, um contendo os tensores
        que serão utilizados para a construção do DataLoader
        de treino da rede neural e outro contendo os tensores
        que serão utilizados para a construção do DataLoader
        de teste da rede neural.
    """
    logger.info('Tranformando os dados para Tensores')
    return ReturnTensor(settings.caminho_treino), ReturnTensor(
        settings.caminho_teste,
    )


def criar_dataloader(train_set, test_set) -> DataLoader:
    """
    Cria os Dataloaders para treinamento da rede.

    Args:
        train_set (_type_): Tensores para o DataLoader de treinamento
        test_set (_type_): Tensores para o DataLoader de teste

    Returns:
        DataLoader: Dois DataLoader, um contendo os tensores
        que serao utilizados para o treino da rede neural e
        outro com os tensores que serão utilizados para teste
        da rede neural.
    """
    logger.info('Criando os Dataloaders')
    train_loader = DataLoader(
        train_set,
        batch_size=settings.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=settings.batch_size,
        shuffle=True,
    )
    logger.info('Dataloaders Criados!')

    return train_loader, test_loader
