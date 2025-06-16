from maxitest.restapi import Api, HttpMethod
from helpers.service import Services
from models.triton.repository_index_v2_response import RepositoryIndexV2Response


class PostRepositoryIndexV2(Api[list[RepositoryIndexV2Response]]):
    """
    POST
    /v2/repository/index

    Retrieve the list of models and their status from the Triton server.
    """

    def __init__(self, headers=None, params=None, body=None):
        super().__init__(
            path="/v2/repository/index",
            method=HttpMethod.POST,
            response_model=RepositoryIndexV2Response,
            service=Services.triton,
            params=params,
            headers=headers,
            body=body,
        )
