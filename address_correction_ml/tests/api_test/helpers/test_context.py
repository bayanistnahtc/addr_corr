import logging


logger = logging.getLogger(__name__)


class TestContext:
    """
    Holds context information for a single test execution
    """
    trace_parent: str | None = None


test_context = TestContext()
