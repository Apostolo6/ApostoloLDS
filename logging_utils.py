"""
Utilitários de logging para o pipeline LiDAR v8.
Fornece um logger estruturado com saída para stdout e ficheiro opcional.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str,
                 log_file: Path = None,
                 level: str = "INFO") -> logging.Logger:
    """
    Cria e configura um logger com formatação consistente.

    Args:
        name: Nome do logger (geralmente nome do módulo/classe).
        log_file: Ficheiro de log opcional. None = só stdout.
        level: Nível de logging ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        logging.Logger configurado.
    """
    log = logging.getLogger(name)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))

    if log.handlers:
        return log  # já configurado (evita handlers duplicados)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)

    log.propagate = False
    return log


def get_logger(name: str) -> logging.Logger:
    """Obtém logger existente ou cria com configuração base."""
    log = logging.getLogger(name)
    if not log.handlers:
        return setup_logger(name)
    return log
