"""
Tests for Representation exceptions.

Author: Ishita Dutta
Project: UIFAF
"""

from representation.exceptions import (
    RepresentationError,
    RepresentationInitializationError,
    InvalidRepresentationFeatureError,
    InvalidRepresentationOutputError,
)


def test_exception_hierarchy():

    assert issubclass(
        RepresentationInitializationError,
        RepresentationError,
    )

    assert issubclass(
        InvalidRepresentationFeatureError,
        RepresentationError,
    )

    assert issubclass(
        InvalidRepresentationOutputError,
        RepresentationError,
    )


def test_raise_base_exception():

    try:
        raise RepresentationError("error")
    except RepresentationError as exc:
        assert str(exc) == "error"