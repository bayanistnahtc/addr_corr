from typing import List, Optional

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """
    Response model for a single corrected address.
    """
    id: str = Field(..., description="Unique identifier matching the request.")
    corrected_address: str = Field(..., description="The corrected address returned by the model.")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class BatchPredictResponse(BaseModel):
    """
    Response model for a batch of corrected address.
    """
    corrected_addresses: List[PredictResponse] = Field(..., description="List of corrected address.")


class HealthResponse(BaseModel):
    """
    Response model fot the health check endpont.
    """
    status: str = Field("OK", description="Service status (OK, DEGRADED, UNHEALTH, ERROR).")
    triton_status: Optional[bool] = Field(None, description="Indicates if Triton server is reachable and model is ready.")
