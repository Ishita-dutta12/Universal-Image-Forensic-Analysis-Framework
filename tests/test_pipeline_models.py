from pathlib import Path

from pipeline.pipeline_models import (
    PipelineRequest,
    PipelineContext,
)


def test_pipeline_request():

    request = PipelineRequest(
        image_path=Path("image.jpg")
    )

    assert request.image_path.name == "image.jpg"


def test_pipeline_context():

    context = PipelineContext()

    assert context.image_info is None

    assert context.reasoning is None