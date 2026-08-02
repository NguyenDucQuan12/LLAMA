from __future__ import annotations

"""
Thiết lập logging thống nhất.
"""

import logging
import sys


def configure_logging(log_level: str) -> None:
    """
    Cấu hình root logger.

    Trong production, bạn có thể thay formatter này bằng JSON formatter
    để gửi log sang Loki, Elasticsearch hoặc hệ thống giám sát khác.
    """

    normalized_level = getattr(
        logging,
        log_level.upper(),
        logging.INFO,
    )

    logging.basicConfig(
        level=normalized_level,
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ),
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
