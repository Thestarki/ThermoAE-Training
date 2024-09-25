"""
Este modulo constroi o autoencoder e executa o treinamento dele.

Ele contém a classe Autoencoder que descreve a topologia
da rede neural, alem de duas funcoes para as etapas de
treinamento e teste da rede.
"""

import sys
from typing import Any

import numpy as np
from torch import nn, no_grad

from configs.AEConfig import ae_settings

sys.path.append('src')


class Autoencoder(nn.Module):
    """
    Uma classe de serviço para construir a topologia do autoencoder.

    Esta classe cria as camadas de entrada, oculta e de saida do
    autoencoder e faz o passo da propagação.

    Args:
        nn (nn.Module): O modulo de redes neurais do Pytorch.

    Returns:
        nn.Module: A arquitetura da rede neural
    """

    def __init__(self) -> None:
        """Topologia do autoencoder."""
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ae_settings.input_size, ae_settings.hidden_size),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(ae_settings.hidden_size, ae_settings.out_size),
        )

    def forward(self, x_tensor) -> Any:
        """Passo de propagação.

        Args:
            x_tensor (tensor): Tensor para treinamento.

        Returns:
            Any: Dados após passarem na rede.
        """
        encoder = self.encoder(x_tensor)
        return self.decoder(encoder)


def treino(
        train_loader,
        rede,
        epoch,
        criterion,
        optimizer,
        device,
        ) -> np.array:
    """
    Esta função executa o treinamento da rede.

    Primeiro a rede é colocada em modo de treinamento,
    então há o passe de propagação, calculo da loss,
    a retropropagação e a atuação do otimizador. Por
    fim se printa a loss da epoca durante o treino.

    Args:
        train_loader (DataLoader): Um Dataloader com os dados para treinamento
        rede (nn.module): Uma classe representando a topologia da rede
        epoch (int): Numero de epocas de treinamento
        criterion (nn.module): Funcao de perda utilizada
        optimizer (torch.optim): Otimizador utilizado
        device (torch.device): ve se dados serao carregados na CPU ou GPU

    Returns:
        np.array: A loss media do treinamento.
    """
    # Setting our network to train mode
    rede.train()

    # List to store all the loss of an epoch
    epoch_loss = []

    for batch in train_loader:

        # Getting the information of the batch
        dado = batch

        # Casting in the gpu
        dado = dado.to(device)

        # Forward pass and calculate the loss
        pred = rede(dado)

        loss = criterion(dado, pred)
        epoch_loss.append(loss.cpu().data)

        # backward pass
        loss.backward()
        optimizer.step()

    epoch_los = np.asarray(epoch_loss)
    print(
        'Train: Epoch %d, Loss: %.4f +/- %.4f'
        % (epoch, epoch_los.mean(), epoch_los.std()),
    )

    return epoch_los.mean()


def teste(
        test_loader,
        rede,
        epoch,
        criterion,
        device,
        ) -> np.array:
    """
    Esta função executa o teste da rede.

    Primeiro a rede é colocada em modo de teste,
    então há o passe de propagação e calculo da loss. Por
    fim se printa a loss da epoca durante o teste.

    Args:
        test_loader (_type_): Um Dataloader com os dados para teste
        rede (_type_): Uma classe representando a topologia da rede
        epoch (_type_): Numero de epocas de teste
        criterion (_type_): Funcao de perda
        device (_type_): indica se os dados serao carregados na CPU ou GPU

    Returns:
        np.array: A loss media do teste.
    """
    rede.eval()

    with no_grad():
        # List to store all the loss of an epoch
        epoch_loss = []

        for batch in test_loader:
            # getting the information of the batch
            dado = batch

            # casting na gpu
            dado = dado.to(device)

            # forward
            pred = rede(dado)
            loss = criterion(dado, pred)

            # Getting the loss
            epoch_loss.append(loss.cpu().data)

        epoch_los = np.asarray(epoch_loss)

        print(
            'Test: Epoch %d, Loss_mean: %.4f +/- %.4f \n'
            % (epoch, epoch_los.mean(), epoch_los.std()),
        )

        return epoch_los.mean()
