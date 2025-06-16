from typing import List, Optional

from pydantic import Field
from models.address_corrector.config_base_model import ConfigBaseModel


class PredictRequest(ConfigBaseModel):
    request_id: str = Field(..., description="Unique identifier for the request.")
    geoparser_address: str = Field(..., description="Data provided by the geoparser.")
    old_address: Optional[str] = Field(None, description="Previously known address.")


class BatchPredictRequest(ConfigBaseModel):
    addresses: List[PredictRequest] = Field(..., description="List of addresses to be corrected")
