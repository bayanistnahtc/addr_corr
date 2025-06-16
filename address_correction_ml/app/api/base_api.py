from fastapi import APIRouter
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    generate_latest,
    multiprocess,
)
from starlette.responses import Response

router = APIRouter()


@router.get("/ping", response_model=dict[str, str])
def ping() -> dict[str, str]:
    """Simple ping endpoint."""
    return {"message": "pong"}


@router.get("/metrics", response_class=Response)
def metrics() -> Response:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
