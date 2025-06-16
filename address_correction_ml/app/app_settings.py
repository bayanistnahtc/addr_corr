import logging
import os
import yaml

from typing import Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


logger = logging.getLogger(__name__)


def get_config(config_path: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration file contents as a dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            if not isinstance(config, dict):
                raise ValueError("Configuration file does not contain a valid dictionary")
            return config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {config_path}") from e


class Settings(BaseSettings):
    config_dir: str = Field(default='../app_configs')
    config_name: str = Field(default='app_config.yaml')
    config: Optional[Dict[str, str]] = {}
    triton_host: str = Field(default='triton')
    triton_port_http: str = Field(default='8000')
    triton_port_grpc: str = Field(default='8001')
    log_level: str = Field(default="INFO")
    triton_request_timeout_seconds: float = Field(default=10.0)
    max_batch_size: int = Field(default=4)

    @property
    def triton_url(self) -> str:
        return f"{self.triton_host}:{self.triton_port_grpc}"
    
    
    # model_config = SettingsConfigDict(
    #     env_file = ".env",
    #     env_file_encodding = "utf-8",
    #     extra="ignore"
    # )



settings = Settings()

try:
    config_file_path = os.path.join(settings.config_dir, settings.config_name)
    settings.config = get_config(config_file_path)
    triton_config = settings.config.get("triton_config")
    settings.triton_host = triton_config.get("host", "localhost")
    settings.triton_port_http = triton_config.get("port", 8000)
    settings.triton_port_grpc = triton_config.get("grpc_port", 8001)
    settings.log_level = triton_config.get("log_level", "INFO")
    settings.triton_request_timeout_seconds = triton_config.get("request_timeout_seconds", 10.0)
    settings.max_batch_size = triton_config.get("max_batch_size", 4)

    logger.info(f"Loaded config from: {config_file_path}")
except Exception as e:
    # Using defaults
    logger.error(f"Counld not load config from: {config_file_path}: {e}") 
    # settings.triton_model_name = "addr_corr_gpt_ensemble"
    # settings.triton_model_version = 1
    # settings.config = {"model_name": settings.triton_model_name}
    