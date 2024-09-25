"""Esse programa executa toda a pipeline do Autoencoder."""

import sys

from loguru import logger
from torch import nn, optim, save

from configs.AEConfig import ae_settings
from configs.DataConfig import data_settings
from configs.Debug_Config import loggerr
from configs.GpuConfig import checar_gpu
from configs.PathConfig import path_settings
from modelo.criar_dataloaders import (chamando_return_tensor, criar_dataloader,
                                      partir_treino_teste)
from modelo.ler_dados import ler_dados_da_db
from modelo.modelo import Autoencoder, teste, treino
from modelo.tratar_dados import filtrar_horas_iniciais, scale_data

sys.path.append('src')
loggerr()

dados = ler_dados_da_db()

dados = dados[data_settings.variables]

dados = filtrar_horas_iniciais(dados)

dados = scale_data(dados)

partir_treino_teste(dados)

train_set, test_set = chamando_return_tensor()

train_loader, test_loader = criar_dataloader(train_set, test_set)

device = checar_gpu()

rede = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adagrad(rede.parameters())


los_treinos = []
los_testes = []

logger.info('Comecando o treinamento!')
for epoch in range(ae_settings.epochs):
    # Calling our train function and getting
    # the loss of the training for plot later
    loss_treino = treino(
        train_loader,
        rede,
        epoch,
        criterion,
        optimizer,
        device,
    )

    loss_teste = teste(
        test_loader,
        rede,
        epoch,
        criterion,
        device,
    )

    los_treinos.append(loss_treino)
    los_testes.append(loss_teste)

logger.info('Treinamento concluido!')
save(rede, path_settings.path_to_model + 'modelo.pt')
