from maxitest.restapi import Api, HttpMethod
from helpers.service import Services
from models.address_corrector.response import PredictResponse, BatchPredictResponse
from models.address_corrector.request import PredictRequest, BatchPredictRequest


class PostV1Predict(Api[PredictResponse]):
    """
    POST
    /api/v1/predict

    Sends a single address for correction.
    """

    def __init__(self, headers=None, params=None, body: PredictRequest | dict = None):
        super().__init__(
            path="/api/v1/predict",
            method=HttpMethod.POST,
            response_model=PredictResponse,
            service=Services.address_corrector,
            params=params,
            headers=headers,
            body=body,
        )


class PostV1BatchPredict(Api[BatchPredictResponse]):
    """
    POST
    /api/v1/batch_predict

    Sends a batch of addresses for correction.
    """

    def __init__(self, headers=None, params=None, body: BatchPredictRequest | dict = None):
        super().__init__(
            path="/api/v1/batch_predict",
            method=HttpMethod.POST,
            response_model=BatchPredictResponse,
            service=Services.address_corrector,
            params=params,
            headers=headers,
            body=body,
        )
