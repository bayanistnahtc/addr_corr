from maxitest.restapi import Api, HttpMethod
from helpers.service import Services


class GetApiPing(Api[None]):
    """
    GET
    /api/v1/ping
    Check service availability.
    """

    def __init__(self, headers=None, params=None, body=None):
        super().__init__(
            path="/api/v1/ping",
            method=HttpMethod.GET,
            response_model=None,
            service=Services.address_corrector,
            params=params,
            headers=headers,
            body=body,
        )
