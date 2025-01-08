import cv2
import os
from pipeline.base import PipelineStep, PipelineImageContainer

class SaveImageStep(PipelineStep):
    def __init__(self, name, output_folder: str = './data/out/processed/', pipeline=None):
        super().__init__(name, pipeline)
        self.output_folder = output_folder

    def process_single(self, input_item: PipelineImageContainer):
        print(f"{input_item.english_label} [{input_item.scientific_label}] - {input_item.page}.v.{input_item.instance} (physical: {input_item.physical_page})")
        
        if not os.path.exists(f"{self.output_folder}{input_item.book}"):
            os.makedirs(f"{self.output_folder}{input_item.book}")
        
        cv2.imwrite(f'{self.output_folder}{input_item.book}/{input_item.page}.jpg', input_item.image)
        
        return None