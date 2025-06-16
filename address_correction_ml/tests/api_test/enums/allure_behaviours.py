from enum import StrEnum


class Allure(StrEnum):
    """
    Reference of epics and features for Allure reports (Address Correction Service)
    """

    # Technical Endpoints Epic
    EPIC_TECH = "Technical Methods"
    FEATURE_METRICS = "Prometheus Metrics"
    FEATURE_PING = "Service Availability Check (Ping)"
    FEATURE_HEALTH = "Service Health Check (Health)"

    # Prediction Service Epic
    EPIC_SERVICE_PREDICT = "Address Correction Service"
    FEATURE_PREDICT_SINGLE = "Single Address Correction"
    FEATURE_PREDICT_BATCH = "Batch Address Correction"
    FEATURE_VALIDATION = "Input Data Validation"

    # Tritton/Model Epic
    EPIC_TRITON = "Triton Inference"
    FEATURE_ML_MODELS = "ML Models on Triton"
