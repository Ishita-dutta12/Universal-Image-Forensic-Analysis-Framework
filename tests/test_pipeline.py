from pathlib import Path

from pipeline.pipeline import UIFAFPipeline

pipeline = UIFAFPipeline()

result = pipeline.analyze(
    Path(r"D:\uifaf\img\iskd.png")
)

print(result)