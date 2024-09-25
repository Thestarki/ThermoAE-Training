"""Configura o Logger."""

from loguru import logger


def loggerr():
    """Criando o Logger."""
    logger.remove()
    logger.add(
        'info.log',
        rotation='1 MB',
        compression='zip',
        retention='1 MINUTE',
        level='DEBUG',
        )
