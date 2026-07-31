"""
UIFAF
Universal Image Forensic Analysis Framework

FastAPI Application
"""
from pathlib import Path
#from unittest import result
from urllib import request

from matplotlib.pylab import rint

from pipeline.pipeline import UIFAFPipeline
from report_generator.report_formatter import ReportFormatter
from fastapi import FastAPI
from typing import Optional

from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from fastapi import HTTPException
import tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from api.file_handler import (
    save_uploaded_image,
    download_image,
    delete_temp_image,
)

from pipeline.pipeline_models import PipelineRequest
from pipeline.pipeline import UIFAFPipeline
from report_generator.report_formatter import ReportFormatter

#import httpx

from tests.test_report_formatter import report
from fastapi.staticfiles import StaticFiles
app = FastAPI(
    title="UIFAF API",
    description="Universal Image Forensic Analysis Framework",
    version="1.0.0",
)
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs",
)
pipeline = UIFAFPipeline()

formatter = ReportFormatter()
@app.get("/")
def root():
    """
    Health check endpoint.
    """

    return {
        "status": "running",
        "framework": "UIFAF",
        "version": "1.0.0",
    }
    
@app.post("/analyze")
async def analyze(
    image_file: UploadFile | None = File(default=None),
    image_url: str | None = Form(default=None),
):
    print("=" * 50)
    print("image_file =", image_file)
    print("image_url  =", image_url)
    """
    Analyze an uploaded image or an external image URL.
    """

    # ----------------------------
    # Validate input
    # ----------------------------
    if image_file is None and image_url is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either an image file or an image URL.",
        )

    if image_file is not None and image_url is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide only one input.",
        )

    image_path = None

    try:

        # ----------------------------
        # Upload from File Explorer
        # ----------------------------
        if image_file is not None:

            image_path = await save_uploaded_image(image_file)

        # ----------------------------
        # External URL
        # ----------------------------
        else:

            image_path = await download_image(image_url)

        # ----------------------------
        # Run pipeline
        # ----------------------------
        request = PipelineRequest(
            image_path=image_path
        )

        print("=" * 50)
        print("Pipeline started")
        print(image_path)
        print("=" * 60)
        print("API DEBUG")
        print("request =", request)
        print("type(request) =", type(request))
        print("PipelineRequest =", PipelineRequest)
        print("isinstance =", isinstance(request, PipelineRequest))
        print("=" * 60)

        result = pipeline.run(request)
        if not result.success:
            raise HTTPException(
        status_code=400,
        detail=result.message,
    )

        
        print("Pipeline finished")
        print(result)
        print(type(result))

        print("Report:")
        print(result.report)

        report = formatter.to_dict(result.report)

        print("Returning JSON")

        return report

    finally:

        if image_path is not None:

            delete_temp_image(image_path)
    