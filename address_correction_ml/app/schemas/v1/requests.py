from typing import List, Optional

from pydantic import BaseModel, Field


class AddressInput(BaseModel):
    """
    Base model for address correction imput data.
    """
    geoparser_address: str = Field(..., description="Data provided by the geoparser (required)")
    old_address: Optional[str] = Field(None, description="Previously known address (optional)")
    

class PredictRequest(AddressInput):
    """
    Request model for single address correction.
    """
    request_id: str = Field(..., description="Unique identifier for the request.")


class BatchPredictRequest(BaseModel):
    """
    Request model for batch address correction.
    """
    addresses: List[PredictRequest] = Field(..., description="List of addresses to be corrected")
