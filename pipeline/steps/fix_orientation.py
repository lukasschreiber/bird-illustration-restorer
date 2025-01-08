from pipeline.base import PipelineStep, PipelineImageContainer
from pymupdf import Document
import cv2

class FixOrientationStep(PipelineStep):
    def __init__(self, name, pipeline=None):
        super().__init__(name, pipeline)

    def process_single(self, input_item: PipelineImageContainer, mask):
        image = input_item
        
        pdf = self.pipeline.get_property("pdf", Document)
        page = pdf[input_item.physical_page - 1]
        
        blocks = page.get_text("blocks")
        
        weighted_count_horizontal = 0
        weighted_count_vertical = 0
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]
            width = x1 - x0
            height = y1 - y0

            if width > height:
                weighted_count_horizontal += width * text.count('\n')
            else:
                weighted_count_vertical += height * text.count('\n')
        
        if weighted_count_vertical > weighted_count_horizontal:
            input_item.image = cv2.rotate(input_item.image, cv2.ROTATE_90_CLOCKWISE)#
    
        return input_item