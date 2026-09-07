from fastapi import FastAPI

from fpvscan.app.routers import health, live, observations, parameters, sessions


def include_test_routers(app: FastAPI) -> None:
    app.include_router(health.router)
    app.include_router(parameters.router)
    app.include_router(sessions.router)
    app.include_router(observations.router)
    app.include_router(live.router)
