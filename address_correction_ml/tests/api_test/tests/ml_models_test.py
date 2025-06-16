from typing import List

import allure
import pytest
from assertpy import soft_assertions

from helpers.assert_that import assert_that
from enums.allure_behaviours import Allure

from api.triton.post_repository_index_v2 import PostRepositoryIndexV2
from models.triton.repository_index_v2_response import RepositoryIndexV2Response, RepositoryState


EXPECTED_ADDRESS_CORRECTION_MODEL = "addr_corr_gpt_ensemble"


@allure.epic(Allure.EPIC_TRITON)
@allure.feature(Allure.FEATURE_ML_MODELS)
class TestMlModels:
    """
    Test for checking ML model status directly on the Triton Inference Server.
    """
    @allure.story("Verify required models are loaded and ready on Triton")
    @pytest.mark.parametrize("expected_model_name", [
        EXPECTED_ADDRESS_CORRECTION_MODEL
    ])
    def test_triton_model_readiness(self, expected_model_name: str):
        """
        Check if specific required model are present and in the READY state
        on the configured Triton server by querying its repository index.
        """

        with allure.step("Request model list from Triton /v2/repository/index"):
            api_call = PostRepositoryIndexV2()
            response = api_call.request()

        with allure.step(f"Verify model: {expected_model_name} is present and READY"), soft_assertions():
            assert_that(response.status_code, f"Triton API Status Code for {expected_model_name} check").is_equal_to(200)
            actual_models: List[RepositoryIndexV2Response] = response.model

            assert_that(actual_models, f"Triton Model List for {expected_model_name} check").is_not_empty()

            found_model = next((model for model in actual_models if model.name == expected_model_name), None)
            assert_that(found_model, f"Model '{expected_model_name}' found on Triton").is_not_none()

            if found_model:
                assert_that(found_model.state, f"State of model '{expected_model_name}'").is_equal_to(RepositoryState.READY)
