from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import app.evaluators
from app.api.routes import dashboard, datasets, evaluations, experiments, metrics, systems
from app.db import models  # noqa: F401
from app.db.models import ImmutableDatasetError
from app.db.session import Base, engine
from app.errors import DomainError


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)  # MVP; switch to Alembic migrations later
    yield


app = FastAPI(title="AI Evaluation Platform - Core Engine", version="0.1.0", lifespan=lifespan)
app.include_router(evaluations.router)
app.include_router(systems.router)
app.include_router(metrics.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(dashboard.router)


@app.exception_handler(DomainError)
async def domain_error_handler(_: Request, exc: DomainError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(ImmutableDatasetError)
async def immutable_handler(_: Request, exc: ImmutableDatasetError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}
