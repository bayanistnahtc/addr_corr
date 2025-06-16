import logging

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette_exporter import PrometheusMiddleware, handle_metrics

from api.base_api import router as base_router
from api.v1.routes import router as predict_router
from app_settings import settings

from core.inference import AddressCorrectionService
from core.model_manager import TritonManager


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for application lifespan events.

    Args:
        app (FastAPI): The FastAPI applicatio instance.
    """
    logger.info('Check dependencies')
    logger.info(f'settings: {settings}')
    logger.info(f'config: {settings.config}')
    triton_config = settings.config.get("triton_config")
    logger.info(f'settings: {triton_config.get("triton_host")} : {triton_config.get("triton_port_http")}')

    app.state.triton_manager = None
    app.state.address_service = None

    try:
        logger.info("Initializing TritonManager...")
        model_name = triton_config["model_name"]
        model_version = triton_config["model_version"]
        request_timeout = int(settings.triton_request_timeout_seconds * 1_000_000)
        triton_manager = TritonManager(
                triton_url=settings.triton_url,
                model_name=model_name,
                model_version=model_version,
                request_timeout=request_timeout
        )

        is_life = triton_manager.check_server_live()
        if is_life:
            is_ready = triton_manager.check_model_ready()
            if not is_ready:
                logger.warning(
                    f"Triton model '{model_name}' version '{model_version or 'latest'}' "
                    "is not ready during startup check."
                )
            else:
                logger.info(
                    f"Initial health check passed: Triton server live and model "
                    f"'{model_name}' version '{model_version or 'latest'}' is ready"
                )
        else:
            logger.error(
                f"Initial health check failed: Triton server '{settings.triton_url}' is not live."
            )
            raise RuntimeError("Triton server unavailable at startup.")
        
        app.state.triton_manager = triton_manager
        logger.info("TritonManager initialized and stored in app.state.")

    except Exception as e:
        logger.exception("Error initializing TritonManager during startup.")
        raise RuntimeError(f"Failed to initialize TritonManager: {e}")
    
    if app.state.triton_manager:
        try:
            logger.info("Initializing AddressCorrectionServise...")
            app.state.address_service = AddressCorrectionService(
                triton_manager=app.state.triton_manager,
                max_batch_size=settings.max_batch_size
            )
            logger.info("AddressCorrectionService initialized and stored in app.state.")
        except Exception as e:
            logger.exception("Error initializing AddressCorrectionService during startup.")
            raise RuntimeError(f"Failed to initialize AddressCorrectionService: {e}")
    else:
        logger.error("Skipping AddressCorrectionService initialization because TritonManager failed.")

    logger.info("Application startup sequence complete. Service is ready to accept requests.")

    yield

    logger.info("Application shutdown sequence initiated...")
    if hasattr(app.state, "triton_manager" ) and app.state.triton_manager:
        app.state.triton_manager.close()
    logger.info("Application shutdown sequence complete.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    PrometheusMiddleware, app_name="name service"
)
app.add_route("/metrics", handle_metrics)

app.include_router(base_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
