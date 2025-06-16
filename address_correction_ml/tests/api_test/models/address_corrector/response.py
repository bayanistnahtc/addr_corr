from typing import List, Optional

from pydantic import Field
from models.address_corrector.config_base_model import ConfigBaseModel


class PredictResponse(ConfigBaseModel):
    id: str = Field(..., description="Unique identifier matching the request.")
    corrected_address: str = Field(..., description="The corrected address returned by the model.")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class BatchPredictResponse(ConfigBaseModel):
    corrected_addresses: List[PredictResponse] = Field(..., description="List of corrected address.")
