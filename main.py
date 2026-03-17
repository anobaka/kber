"""KBER – Enterprise Knowledge Base Management System.

Main entry point: initializes database, starts scheduler, and launches the Feishu bot.
"""

import logging
import os
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
    # Alembic's fileConfig resets root logger – restore level and format.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in root_logger.handlers:
        h.setFormatter(formatter)

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
