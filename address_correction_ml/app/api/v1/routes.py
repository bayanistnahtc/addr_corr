import logging
import traceback
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status
)

from core.inference import AddressCorrectionService
from core.model_manager import TritonManager
from dependency import get_address_correction_service, get_triton_manager
from .prom_metrics import (
    FAILURE_COUNT,
    EXCEPTION_COUNTER,
    HEALTH_CHECK_COUNTER,
    PREDICT_TIME,
    RESPONSE_PAYLOAD_SIZE,
    REQUEST_COUNT,
    REQUEST_PAYLOAD_SIZE,
    SUCCESS_COUNT
)
from schemas.v1.requests import BatchPredictRequest, PredictRequest
from schemas.v1.responses import (
    BatchPredictResponse,
    HealthResponse,
    PredictResponse
) 
    

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
        "/predict", 
        description="Predict", 
        response_model=PredictResponse
)
async def predict_route(
    request: PredictRequest,
    service: AddressCorrectionService = Depends(get_address_correction_service)
    ) -> PredictResponse:
    """
    Handle POST request to correct a single addres using the prediction model.

    Args:
        request (PredictRequest): Request object containing an address inputs.
        service (AddressCorrectionService): The injected instance of the core address correction service logic.
        
    Returns
        PredictResponse: The result of correction.

    Raises
        HTTPException: If prediction fails, returns error with details.

    """
    endpoint_label = "/predict"
    method_label = "predict_route"

    logger.info(f"Received single request: id={request.request_id}")
    REQUEST_COUNT.labels(method=method_label, endpoint=endpoint_label).inc()

    result: Optional[PredictResponse] = None
    
    try:
        with PREDICT_TIME.labels(
            method=method_label,
            endpoint=endpoint_label
        ).time():
            result = await service.correct_single(request)

        if result.error_message:
            logger.warning(f"Single predictions failed for id={request.request_id}")
        SUCCESS_COUNT.labels(method=method_label, endpoint=endpoint_label).inc()
        return result

    except Exception as e:
        logger.exception(f"Unhandled error during single correction for id={request.request_id}", exc_info=e)
        logger.error(traceback.format_exc())
        EXCEPTION_COUNTER.labels(method=method_label, endpoint=endpoint_label).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Internal server error processing request: {type(e).__name__}"
        )


@router.post(
        "/batch_predict",
        description="Predict Batch", 
        response_model=BatchPredictResponse
)
async def batch_predict_route(
    request: BatchPredictRequest,
    service: AddressCorrectionService = Depends(get_address_correction_service)
) -> BatchPredictResponse:
    """
    Handles POST request to correct a batch of addresses using the prediction model.

    Args:
        request (BatchPredictRequest): Request object containing a list of address inputs.
        service (AddressCorrectionService): The injected instance of the core address correction service logic.

    Returns
        BatchPredictResponse: Response: A response object with a list of corrected addresses.

    Raises
        HTTPException: If prediction fails, returns error with details.

    """
    endpoint_label = "/batch_predict"
    method_label = "batch_predict_route"

    num_requests = len(request.addresses)
    logger.info(f"Received batch request: with {num_requests} items.")

    if not request.addresses:
        logger.info("Received empty batch request, returning empty response.")
        return BatchPredictResponse(corrected_addresses=[])

    results: Optional[List[PredictResponse]] = None
    try:
        with PREDICT_TIME.labels(
            method=method_label,
            endpoint=endpoint_label
        ).time():
            results = await service.correct_batch(request.addresses)
            results = BatchPredictResponse(corrected_addresses=results)
        SUCCESS_COUNT.labels(method=method_label, endpoint=endpoint_label).inc()
        return results

    except Exception as e:
        logger.exception(f"Unhandled error during batch correction ({num_requests})", exc_info=e)
        logger.error(traceback.format_exc())
        EXCEPTION_COUNTER.labels(method=method_label, endpoint=endpoint_label).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Internal server error processing batch: {type(e).__name__}"
        )
    

@router.get(
    "/health",
    description="Checks the connection to the Triton Inference Server",
    response_model=HealthResponse
)
async def health_check(
    response: Response,
    triton_manager: TritonManager = Depends(get_triton_manager)
) -> HealthResponse:
    """
    Handles GET requests to perform a health check.
    
    Args: 
        response (Response): FastAPI Response object.
        triton_manager (TritonManager): The injected instance of the TritonManager.
    
    Returns:
        HealthResponse: An object indicating the overall status and Triton model readiness.
    """
    endpoint_label = "/health"
    method_label = "health_check"
    logger.debug("Health check endpoint requested.")
    HEALTH_CHECK_COUNTER.labels(method=method_label, endpoint=endpoint_label).inc()

    triton_live = False
    model_ready = False
    overall_status = "ERROR"
    
    try:
        triton_live = await triton_manager.check_server_live()
        if triton_live:
            try:
                model_ready = await triton_manager.check_model_ready()
            
                overall_status = "OK" if triton_live and model_ready else "DEGRADED"
                logger.debug(f"Health Check: Triton Live: {triton_live}, Model Ready: {model_ready}, Status: {overall_status}")
            except Exception as e:
                logger.warning(f"Healthh Check: Triton is live, but model check failed: {e}", exc_info=True)
                overall_status = "DEGRADED"
        else:
            logger.warning(f"Healthh Check: Triton is not live", exc_info=True)
            overall_status = "UNHEALTHY"

    except ConnectionError as e:
        logger.error(f"Health Check: Connection error during Triton check: {e}", exc_info=True)
        overall_status = "UNHEALTHY"
    except Exception as e:
        logger.exception(f"Health Check: Unexpected error: {e}", exc_info=True)
        overall_status = "ERROR"

    return HealthResponse(
            status=overall_status,
            triton_status=model_ready
        )
