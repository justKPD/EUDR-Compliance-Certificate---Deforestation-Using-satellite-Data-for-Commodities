"""
Domain-specific exceptions for EUDR compliance system.
"""


class EUDRBaseError(Exception):
    """Base class for all EUDR errors."""


class DataAcquisitionError(EUDRBaseError):
    """Raised when satellite data cannot be fetched from GEE."""


class PreprocessingError(EUDRBaseError):
    """Raised when image preprocessing fails."""


class ModelInferenceError(EUDRBaseError):
    """Raised when a model forward pass fails."""


class ComplianceEngineError(EUDRBaseError):
    """Raised for errors in compliance logic."""


class ReportGenerationError(EUDRBaseError):
    """Raised when PDF report generation fails."""
