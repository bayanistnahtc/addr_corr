from pydantic import Field
from models.address_corrector.config_base_model import ConfigBaseModel


class ValidationError(ConfigBaseModel):
    loc: list[str] = Field(alias='loc')
    msg: str = Field(alias='msg')
    type_key: str = Field(alias='type')
    