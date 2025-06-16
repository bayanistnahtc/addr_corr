import allure
import asyncio
import logging
import pytest
from pathlib import Path

from helpers.test_context import test_context
from helpers import generator


BASE_DIR = Path(__file__).resolve().parent


def pytest_configure():
    """
    Configure logging for test runs.
    """
    log_filename = BASE_DIR.joinpath('api_test.log')

    logging.basicConfig(
        filename=log_filename,
        format='%(asctime)s.%(msecs)04d [%(levelname)8s] %(name)-20s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level='INFO',
        filemode='w',
    )
    logger = logging.getLogger(__name__)
    logger.info(f"--- Test run starting, logging to {log_filename} ---")


@pytest.fixture(scope="session")
def event_loop():
    """
    Provedes a session-scoped asyncio event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError: 
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@allure.title("TestContext")
@pytest.fixture(scope='function', autouse=True)
def set_test_context(request):
    """
    Sets up and cleans the test context for each test function.
    """
    test_context.trace_parent = generator.get_trace_parent()
    logging.info(f"Starting test '{request.node.name}' with trace_parent: {test_context.trace_parent}")
    yield
    logging.info(f"Finishing test '{request.node.name}'. Clearing")
    test_context.trace_parent = None
