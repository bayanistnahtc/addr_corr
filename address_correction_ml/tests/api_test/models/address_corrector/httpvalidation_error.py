from typing import List
from pydantic import Field

from models.address_corrector.config_base_model import ConfigBaseModel
from models.address_corrector.validation_error import ValidationError
 

class HTTPValidationError(ConfigBaseModel):
    detail: List[ValidationError] = Field(alias='detail')
