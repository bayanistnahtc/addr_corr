from typing import Any, Dict, Optional, Tuple
from helpers.test_context import test_context
from helpers.config import configuration
from maxitest.restapi import Service


class TritonService(Service):
    """
    Triton Inference Server
    """

    def __init__(self, url: str):
        super().__init__(
            url=url
        )

    def get_default_params(
            self
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, any]]]:
        """
        Provide default parameters for requests to Triton.
        """
        headers = {
            'traceparent': test_context.trace_parent
        }
        params = {}
        body = {}

        return headers, params, body


class AddressCorrectorService(Service):
    """
    Service definition for the Address Correction API under test.
    """

    def __init__(self, url: str):
        super().__init__(
            url=url,
            timeout=10
        )

    def get_default_params(
            self
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, any]]]:
        """
        Provide default parameters for requests to the Address Correction service..
        """
        headers = {
            'traceparent': test_context.trace_parent
        }
        params = {}
        body = {}

        return headers, params, body


class Services:
    """
    Provides instances of the configured services.
    """
    triton = TritonService(url=configuration.service.triton)
    address_corrector = AddressCorrectorService(url=configuration.service.address_corrector)
