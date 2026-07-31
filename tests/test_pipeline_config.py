from pipeline.pipeline_config import PipelineConfig


def test_default_config():

    config = PipelineConfig()

    config.validate()

    assert config.batch_size == 1

    assert config.output_directory.exists()