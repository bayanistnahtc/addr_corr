import allure

from assertpy import soft_assertions

from api.address_corrector.post_v1_predict import (
    BatchPredictRequest,
    PostV1Predict,
    PostV1BatchPredict,
    PredictRequest
)
from models.address_corrector.response import PredictResponse, BatchPredictResponse
from helpers.assert_that import assert_that
from helpers.generator import generate_address_input
from enums.allure_behaviours import Allure


@allure.epic(Allure.EPIC_SERVICE_PREDICT)
@allure.feature(Allure.FEATURE_PREDICT_SINGLE)
class TestPredictorSingle:
    """
    Tests for the single prediction endpoint /api/v1/predict.
    """

    @allure.story("Successful single prediction")
    def test_predict_single_success(self):
        """
        Test a successful prediction from a single input.
        """

        with allure.step("Prepare valid request body"):
            single_input = generate_address_input(id_prefix="test_ok")
            request_body = PredictRequest(
                **single_input
            )

        with allure.step("Send POST request to /api/v1/predict"):
            api_call = PostV1Predict(body=request_body)
            response = api_call.request()

        with allure.step("Verify successful response:"), soft_assertions():
            assert_that(response.status_code, "HTTP Status Code").is_equal_to(200)

            response_obj: PredictResponse = response.model
            assert_that(response_obj.id, "Response ID matches request ID").is_equal_to(request_body.request_id)
            assert_that(response_obj.corrected_address, "Corrected Address").is_not_none().is_not_empty()
            assert_that(response_obj.error_message, "Error Message").is_none()


@allure.epic(Allure.EPIC_SERVICE_PREDICT)
@allure.feature(Allure.FEATURE_PREDICT_BATCH)
class TestPredictorBatch:
    """
    Tests for the batch prediction endpoint /api/v1/batch_predict.
    """

    @allure.story("Successful batch prediction")
    def test_predict_batch_success(self):
        """
        Test a successful batch prediction fro a single input.
        """

        with allure.step("Prepare valid batch request body"):
            batch_input = {"addresses": [
                generate_address_input(id_prefix="test_ok_1"), 
                generate_address_input(id_prefix="test_ok_2", empty_old_address=True)
            ]}
            request_body = BatchPredictRequest(
                **batch_input
            )

        with allure.step("Send POST request to /api/v1/batch_predict"):
            api_call = PostV1BatchPredict(body=request_body)
            response = api_call.request() # expects 200

        with allure.step("Verify successful batch response:"), soft_assertions():
            assert_that(response.status_code, "HTTP Status Code").is_equal_to(200)

            response_obj: BatchPredictResponse = response.model
            assert_that(response_obj.corrected_addresses, "Batch Responses List").is_length(len(request_body.addresses))

            results_map = {res.id: res for res in response_obj.corrected_addresses}
            original_map = {req.request_id: req for req in request_body.addresses}

            for req_id, original_req in original_map.items():
                assert_that(results_map, f"Result for ID '{req_id}").contains_key(req_id)

                item_result = results_map[req_id]
                assert_that(item_result.id, f"Item {req_id} - Response ID").is_equal_to(original_req.request_id)

                if "fail" in original_req.geoparser_address:
                    assert_that(item_result.corrected_address, f"Item {req_id} - Corrected Address (expected fail)").is_none()
                    assert_that(item_result.error_message, f"Item {req_id} - Error Message (expected fail)").is_not_none().is_not_empty()
                else:
                    assert_that(item_result.corrected_address, f"Item {req_id} - Corrected Address (expected success)").is_not_none().is_not_empty()
                    assert_that(item_result.error_message, f"Item {req_id} - Error Message (expected success)").is_none()
