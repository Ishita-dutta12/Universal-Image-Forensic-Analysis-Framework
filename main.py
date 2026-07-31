from pathlib import Path

from sklearn.pipeline import Pipeline

from pipeline.pipeline import UIFAFPipeline
from pipeline.pipeline_models import PipelineRequest


def main():

    pipeline = UIFAFPipeline()

    request = PipelineRequest(
        image_path=Path("img/test.jpeg")
    )

    result = pipeline.run(request)
    print(pipeline.backbone.backbone.model)
    print("=" * 80)
    print(result.message)
    print("=" * 80)

    report = result.report

    print(report)

    print("=" * 80)
    print(report.confidence_percentage)
    print(report.risk_percentage)


if __name__ == "__main__":
    main()