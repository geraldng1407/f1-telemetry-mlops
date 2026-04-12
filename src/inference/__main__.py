"""Allow ``python -m src.inference`` to start the uvicorn server."""

from __future__ import annotations

import uvicorn

from src.inference.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "src.inference.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
