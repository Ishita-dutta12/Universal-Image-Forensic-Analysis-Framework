import numpy as np

from visualization.heatmap import HeatmapGenerator


def test_heatmap_generation():

    cam = np.random.rand(224,224).astype(np.float32)

    generator = HeatmapGenerator()

    result = generator.generate(cam)

    assert result.heatmap.shape == (224,224,3)