"""
For generating test data
"""
import random
import string
import uuid


def get_trace_parent() -> str:
    """
    Generate traceparent
    """
    trace_id = uuid.uuid4().hex[:32]
    span_id = uuid.uuid4().hex[:16]

    trace_parent = f"00-{trace_id}-{span_id}-01"

    return trace_parent


def random_string(length: int = 10) -> str:
    """
    Generates a random alphanumeric string
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_address_input(id_prefix: str = "test", address_len: int = 20, empty_old_address: bool = False) -> dict:
    """
    Generates a sample valid adress input dict.
    """

    return {
        "request_id": f"{id_prefix}-{random_string(6)}",
        "geoparser_address": f"Geoparser Address {random_string(address_len)}",
        "old_address": f"Old Address {random_string(address_len)}" if not empty_old_address else None
    }
