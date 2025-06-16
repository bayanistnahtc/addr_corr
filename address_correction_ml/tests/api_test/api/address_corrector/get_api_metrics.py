from maxitest.restapi import Api, HttpMethod
from helpers.service import Services


class GetApiMetrics(Api[None]):
    """
    GET
    /api/v1/metrics
    Retrieve Prometheus metrics.
    """

    def __init__(self, headers=None, params=None, body=None):
        super().__init__(
            path="/api/v1/metrics",
            method=HttpMethod.GET,
            response_model=None,
            service=Services.address_corrector,
            params=params,
            headers=headers,
            body=body,
        )
