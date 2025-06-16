from pathlib import Path
from pydantic import BaseModel

import json
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_FILE = BASE_DIR.joinpath('configuration.json')


class BaseConfigFile(BaseModel, extra='forbid'):
    pass


class ServiceConfig(BaseConfigFile):
    triton: str
    address_corrector: str


class ConfigFile(BaseConfigFile):
    """
    Structure of the test configuration file (configuration.json)
    """
    service: ServiceConfig


with open(CONFIG_FILE, encoding='utf-8-sig') as file:
    config_file = file.read()

config_json = json.loads(config_file)
configuration = ConfigFile(**config_json)
