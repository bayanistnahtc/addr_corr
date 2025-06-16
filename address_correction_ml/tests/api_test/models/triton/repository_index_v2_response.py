from enum import StrEnum
from models.triton.config_base_model import ConfigBaseModel


class RepositoryState(StrEnum):
    READY = "READY"


class RepositoryIndexV2Response(ConfigBaseModel):
    name: str
    version: int = None
    state: RepositoryState = None
