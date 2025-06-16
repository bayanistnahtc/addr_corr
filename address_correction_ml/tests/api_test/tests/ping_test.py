import allure
from assertpy import soft_assertions

from api.address_corrector.get_api_ping import GetApiPing
from helpers.assert_that import assert_that
from enums.allure_behaviours import Allure


@allure.epic(Allure.EPIC_TECH)
@allure.feature(Allure.FEATURE_PING)
class TestPing:

    @allure.story("Check service availability via ping")
    @allure.title("Test GET /api/v1/ping endpoint")
    def test_ping(self):

        with allure.step("Send GET request to /api/v1/ping"):
            response = GetApiPing().request()

        with allure.step("Verify response status code"), soft_assertions():
            assert_that(response.status_code, "HTTP Status Code").is_equal_to(200)
