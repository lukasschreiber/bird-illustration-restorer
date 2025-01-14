import cv2
import os
from pipeline.utils import PipelineImageContainer

def save_pipeline_output(
    directory: str,
    image: PipelineImageContainer,
):
    print(f"{image.english_label} [{image.scientific_label}] - {image.page}.v.{image.instance} (physical: {image.physical_page})")
        
    if not os.path.exists(f"{directory}{image.book}"):
        os.makedirs(f"{directory}{image.book}")
    
    cv2.imwrite(f'{directory}{image.book}/{image.page}.jpg', image.image)