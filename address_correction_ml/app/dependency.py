import logging
from typing import Optional

from fastapi import Request, HTTPException, status

from core.model_manager import TritonManager
from core.inference import AddressCorrectionService


logger = logging.getLogger(__name__)


def get_triton_manager(request: Request) -> TritonManager:
    """
    Dependancy injector for the applications's TritonManager instance.

    Args:
        request (Request): The incoming FastAPI request object, used to access 'app.state'
    
    Returns:
        TritonManager: The shared TritonManager instance.
    
    Raises:
        HTTPException(503): If the TritonManager instance is not found.
    """

    triton_manager: Optional[TritonManager] = getattr(
        request.app.state, 'triton_manager', None
    )

    if not triton_manager:
        logger.error("TritonManager not found in app.state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service dependency 'TritonManager' is not available"
        )
    return triton_manager


def get_address_correction_service(request: Request) -> AddressCorrectionService:
    """
    Dependancy inhector for the application's AddressCorrectionService instance.
    
    Args:
        request (Request): The incoming FastAPI request object, used to access 'app.state'.
    
    Returns:
        AddressCorrectionService: The shared AddressCorrectionService instance.
    
    Raises:
        HTTPException(503): If the AddressCorrectionService instance is not found.
    """
    service: Optional[AddressCorrectionService] = getattr(
        request.app.state, 'address_service', None
    )

    if not service:
        logger.error("AddressCorrectionService not found in app.state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service dependency 'AddressCorrectionService' is not available"
        )
    return service
