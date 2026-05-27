"""FastAPI entrypoint for CoachVision."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from coachvision.api.router import api_router
from coachvision.core.config import settings
from coachvision.db.bootstrap import bootstrap_database


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()


@app.on_event("startup")
def _on_startup() -> None:
    bootstrap_database()
