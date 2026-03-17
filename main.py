"""KBER – Enterprise Knowledge Base Management System.

Main entry point: initializes database, starts scheduler, and launches the Feishu bot.
"""

import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler

# Ensure log directory exists
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_log_fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
_log_datefmt = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    datefmt=_log_datefmt,
)

# File handler – 10 MB per file, keep 5 backups
_file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "kber.log"),
    maxBytes=100 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(fmt=_log_fmt, datefmt=_log_datefmt))
_file_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_file_handler)
logger = logging.getLogger("kber")


def main() -> None:
    # Ensure repos directory exists
    from app.config import config
    os.makedirs(config.REPOS_BASE_DIR, exist_ok=True)

    # Run database migrations (Alembic)
    logger.info("Running database migrations...")
    from alembic import command
    from alembic.config import Config as AlembicConfig
    alembic_cfg = AlembicConfig("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    # Alembic's fileConfig resets root logger – restore level, format, and file handler.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=_log_fmt, datefmt=_log_datefmt)
    for h in root_logger.handlers:
        h.setFormatter(formatter)
    # Re-add file handler if Alembic removed it
    if _file_handler not in root_logger.handlers:
        root_logger.addHandler(_file_handler)

    # Connect to Milvus
    logger.info("Connecting to Milvus...")
    from app.services.milvus_service import milvus_service
    milvus_service.connect()

    # Start scheduler
    logger.info("Starting scheduler...")
    from app.scheduler.tasks import start_scheduler, stop_scheduler
    start_scheduler()

    # Graceful shutdown
    def _shutdown(sig: int, frame: object) -> None:
        logger.info("Shutting down...")
        stop_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start Feishu bot (blocking – runs WebSocket event loop)
    logger.info("Starting Feishu bot...")
    from app.bot.handler import feishu_bot
    feishu_bot.start(message_handler=None)


if __name__ == "__main__":
    main()
