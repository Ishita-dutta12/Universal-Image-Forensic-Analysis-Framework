"""
Tests for UIFAF Feature Extractor.

Author: Ishita Dutta
Project: UIFAF
"""

import numpy as np
import pytest
import torch
from PIL import Image

from preprocessing.decoded_image import DecodedImage
from preprocessing.image_info import ImageInfo

from backbone.feature_extractor import FeatureExtractor
from backbone.backbone_models import BackboneOutput
from backbone.exceptions import BackboneError


def create_decoded_image(
    width: int = 224,
    height: int = 224,
) -> DecodedImage:
    """
    Create a dummy DecodedImage.
    """

    rgb = np.random.randint(
        0,
        256,
        (height, width, 3),
        dtype=np.uint8,
    )

    pil = Image.fromarray(rgb)

    info = ImageInfo(
        filename="dummy.png",
        image_format="PNG",
        width=width,
        height=height,
        channels=3,
        color_mode="RGB",
        bit_depth=8,
        file_size=1024,
        has_exif=False,
        exif_orientation=None,
    )

    return DecodedImage(
        rgb_image=rgb,
        original_image=pil,
        info=info,
        filepath=None,
        validation_passed=True,
    )


def test_feature_extractor_initialization():
    """
    Feature extractor should initialize correctly.
    """

    extractor = FeatureExtractor(
        pretrained=False,
        device="cpu",
    )

    assert extractor.device.type == "cpu"


def test_prepare_tensor():
    """
    Image should become normalized BCHW tensor.
    """

    extractor = FeatureExtractor(
        pretrained=False,
        device="cpu",
    )

    decoded = create_decoded_image()

    tensor = extractor._prepare_tensor(decoded)

    assert isinstance(tensor, torch.Tensor)

    assert tensor.shape == (1, 3, 224, 224)

    assert tensor.dtype == torch.float32


def test_forward():
    """
    Forward should return BackboneOutput.
    """

    extractor = FeatureExtractor(
        pretrained=False,
        device="cpu",
    )

    decoded = create_decoded_image()

    output = extractor(decoded)

    assert isinstance(output, BackboneOutput)

    assert output.num_stages == 4


def test_original_size():
    """
    Original image size should be preserved.
    """

    extractor = FeatureExtractor(
        pretrained=False,
        device="cpu",
    )

    decoded = create_decoded_image(
        width=320,
        height=240,
    )

    output = extractor(decoded)

    assert output.original_size == (240, 320)


def test_invalid_dimensions():
    """
    Invalid image dimensions should fail.
    """

    extractor = FeatureExtractor(
        pretrained=False,
        device="cpu",
    )

    decoded = create_decoded_image()

    decoded.rgb_image = np.random.rand(224, 224)

    with pytest.raises(BackboneError):
        extractor._prepare_tensor(decoded)


def test_invalid_channels():
    """
    Non-RGB images should fail.
    """

    extractor = FeatureExtractor(
        pretrained=False,
        device="cpu",
    )

    decoded = create_decoded_image()

    decoded.rgb_image = np.random.rand(
        224,
        224,
        1,
    )

    with pytest.raises(BackboneError):
        extractor._prepare_tensor(decoded)


def test_feature_shapes():
    """
    Feature maps should decrease spatially.
    """

    extractor = FeatureExtractor(
        pretrained=False,
        device="cpu",
    )

    decoded = create_decoded_image()

    output = extractor(decoded)

    stages = output.stages

    heights = [
        stage.height
        for stage in stages
    ]

    widths = [
        stage.width
        for stage in stages
    ]

    assert heights == sorted(
        heights,
        reverse=True,
    )

    assert widths == sorted(
        widths,
        reverse=True,
    )


def test_feature_channels():
    """
    FeatureMap metadata should match tensor channels.
    """

    extractor = FeatureExtractor(
        pretrained=False,
        device="cpu",
    )

    decoded = create_decoded_image()

    output = extractor(decoded)

    for feature in output.stages:

        assert (
            feature.tensor.shape[1]
            == feature.channels
        )