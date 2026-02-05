import logging

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when dataset schema validation fails."""
    pass


def validate_schema(df, required_columns, name: str):
    missing = set(required_columns) - set(df.columns)

    if missing:
        logger.error(
            "%s schema validation failed. Missing columns: %s",
            name,
            sorted(missing),
        )
        raise SchemaError(
            f"{name} is missing required columns: {sorted(missing)}"
        )

    logger.info(
        "%s schema validation passed (%d columns)",
        name,
        len(required_columns),
    )
