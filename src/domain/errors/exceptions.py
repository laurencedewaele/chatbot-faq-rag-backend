class DomainError(Exception):
    """Base domain exception."""

class EmptyQuestionError(DomainError):
    """Raised when a user asks an empty question."""

class InvalidThresholdsError(DomainError):
    """Raised when the FAQ judge threshold is greater than the FAQ accept threshold."""
