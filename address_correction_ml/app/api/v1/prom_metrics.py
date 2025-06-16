# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge


PREDICT_TIME = Histogram(
    name="predict_time_seconds", 
    documentation="Time spent processing predicttions",
    labelnames=("method", "endpoint")
)
EXCEPTION_COUNTER = Counter(
    name="prediction_exceptions_total", 
    documentation="Number of exceptions during prediction",
    labelnames=("method", "endpoint")
)
HEALTH_CHECK_COUNTER = Counter(
    name="health_checks_total",
    documentation="Number of health check calls",
    labelnames=("method", "endpoint")
)
REQUEST_COUNT = Counter(
    name="request_count_total", 
    documentation="Total number of incoming requests",
    labelnames=("method", "endpoint")
)
REQUEST_PAYLOAD_SIZE = Histogram(
    name="request_payload_bytes",
    documentation="Size of the request payload in bytes",
    labelnames=("method", "endpoint")
)
RESPONSE_PAYLOAD_SIZE = Histogram(
    name="response_payload_bytes",
    documentation="Size of the response payload in bytes",
    labelnames=("method", "endpoint")
)
SUCCESS_COUNT = Counter(
    name="prediction_success_total", 
    documentation="Number of succsessfull predictions",
    labelnames=("method", "endpoint")
)
FAILURE_COUNT = Counter(
    name="prediction_failure_total", 
    documentation="Number of failed predictions",
    labelnames=("method", "endpoint")
)
# MODEL_LOADED = Gauge(
#     name="model_loaded_flag",
#     documentaion="Flag indicatinf wherher the ML model is loaded (1) or not (0)"
# )
