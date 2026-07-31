
import numpy as np

from visualization.overlay import OverlayGenerator


def test_overlay_generation():

    image = np.zeros((224,224,3), dtype=np.uint8)

    heatmap = np.ones((224,224,3), dtype=np.uint8) * 255

    generator = OverlayGenerator()

    result = generator.generate(
        image=image,
        heatmap=heatmap,
    )

    assert result.overlay.shape == image.shape