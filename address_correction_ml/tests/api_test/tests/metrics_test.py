import allure

from assertpy import soft_assertions

from api.address_corrector.get_api_metrics import GetApiMetrics
from helpers.assert_that import assert_that
from enums.allure_behaviours import Allure


@allure.epic(Allure.EPIC_TECH)
@allure.feature(Allure.FEATURE_METRICS)
class TestMetrics:

    @allure.story("Check metrics endpoint availability and format")
    def test_metrics(self):
        """
        Send a request to the /metrics enpoint.
        Verify the status code id 200 OK and the Content-type is correct
        for Prometheus metrics.
        Does not validate specidic metric values.
        """

        with allure.step("Send GET request to /metrics"):
            response = GetApiMetrics().request()

        with allure.step(f"Verify response status and headers"), soft_assertions():
            assert_that(response.status_code, "HTTP Status Code").is_equal_to(200)


            headers_lower = {k.lower(): v for k, v in response.headers.items()}
            content_type = headers_lower.get('content-type', '')
            assert_that(content_type, 'Content-Type Header').contains('text/plain')
            assert_that(content_type, 'Content-Type Encoding (utf-8)').contains('utf-8')
            
