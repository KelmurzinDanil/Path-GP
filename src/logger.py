import logging
import sys

def setup_logger(name: str = "GP-Feynman", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Формат: [INFO] Имя_Модуля: Сообщение
        formatter = logging.Formatter(
            fmt='[%(levelname)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False 
    return logger