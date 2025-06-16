import logging
import allure
from assertpy import soft_fail, assert_that as at
from assertpy.assertpy import AssertionBuilder


logger = logging.getLogger(__name__)


def assert_that(val, description) -> AssertionBuilder:
    """
    Wraps assertpy's assert_that to automatically create an Allure step for each assertion.
    """
    with allure.step(description):
        return at(val, description)
