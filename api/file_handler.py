from __future__ import annotations

import uuid
from pathlib import Path

import httpx
from fastapi import HTTPException, UploadFile

# Project temp directory
TEMP_DIR = Path("outputs") / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
}
async def save_uploaded_image(
    image_file: UploadFile,
) -> Path:

    extension = Path(image_file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format."
        )

    filename = f"upload_{uuid.uuid4().hex}{extension}"

    image_path = TEMP_DIR / filename

    contents = await image_file.read()

    image_path.write_bytes(contents)

    return image_path

async def download_image(
    image_url: str,
) -> Path:

    try:

        async with httpx.AsyncClient(timeout=30) as client:

            response = await client.get(image_url)

            response.raise_for_status()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to download image."
        )

    content_type = response.headers.get("content-type", "")

    if not content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="URL does not point to an image."
        )

    extension = Path(image_url).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        extension = ".jpg"

    filename = f"url_{uuid.uuid4().hex}{extension}"

    image_path = TEMP_DIR / filename

    image_path.write_bytes(response.content)

    return image_path

def delete_temp_image(
    image_path: Path,
) -> None:

    try:

        if image_path.exists():

            image_path.unlink()

    except Exception:

        pass