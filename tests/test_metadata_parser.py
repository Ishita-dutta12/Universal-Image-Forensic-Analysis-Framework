from metadata.metadata_models import MetadataInfo
from metadata.metadata_parser import MetadataParser


def test_parse_basic_metadata():

    raw = {
        "Make": "Canon",
        "Model": "EOS R6",
        "Software": "Adobe Photoshop",
        "Orientation": 1,
        "DateTimeOriginal": "2026:07:22 12:30:00",
    }

    metadata = MetadataParser.parse(raw)

    assert isinstance(metadata, MetadataInfo)

    assert metadata.camera.make == "Canon"

    assert metadata.camera.model == "EOS R6"

    assert metadata.camera.software == "Adobe Photoshop"

    assert metadata.capture.orientation == 1

    assert metadata.capture.datetime_original == "2026:07:22 12:30:00"