from pydantic import BaseModel, ConfigDict


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        populate_by_name=True,
    )
