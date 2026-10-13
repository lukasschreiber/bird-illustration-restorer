import cv2
import os
import re
from pipeline.utils import PipelineImageContainer


def output_filename(image: PipelineImageContainer) -> str:
    name = re.sub(r"[^\w-]+", "-", str(image.english_label), flags=re.UNICODE).strip("-_")
    name = name[:120].rstrip("-_") or "Unknown-bird"
    instance = f"-v{image.instance}" if image.instance else ""
    return f"{image.page}{instance}-{name}.jpg"

def save_pipeline_output(
    directory: str,
    image: PipelineImageContainer,
):
    if not os.path.exists(f"{directory}{image.book}"):
        os.makedirs(f"{directory}{image.book}")
    
    path = os.path.join(directory, image.book, output_filename(image))
    if not cv2.imwrite(path, image.image):
        raise OSError(f"Could not save processed image: {path}")
    print(f"Saved: {path}", flush=True)
