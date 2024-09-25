"""Função que checa se tem GPU disponivel."""

import torch


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
