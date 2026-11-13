"""Shared structured logger. Import `logger` anywhere in the app."""
import sys

from loguru import logger

from app.config import get_settings

_settings = get_settings()

logger.remove()
logger.add(
    sys.stderr,
    level=_settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    backtrace=False,
    diagnose=False,
)

__all__ = ["logger"]
