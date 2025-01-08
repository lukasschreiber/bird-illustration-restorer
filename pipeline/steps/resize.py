import cv2
from pipeline.base import PipelineStep, PipelineImageContainer

class ResizeStep(PipelineStep):
    def __init__(self, name, width: int|None = None, height: int|None = None, max: int|None = None, pipeline=None):
        super().__init__(name, pipeline)
        self.size = (width, height)
        self.max = max

    def process_single(self, input_item: PipelineImageContainer, mask):
        if self.max is not None:
            h, w = input_item.image.shape[:2]
            aspect_ratio = w / h
            if w > h:
                new_w = self.max
                new_h = int(self.max / aspect_ratio)
            else:
                new_h = self.max
                new_w = int(self.max * aspect_ratio)
            
            input_item.image = cv2.resize(input_item.image, (new_w, new_h))
            
        elif self.size[0] is not None and self.size[1] is not None:
            input_item.image = cv2.resize(input_item.image, self.size)
        elif self.size[0] is not None:
            input_item.image = cv2.resize(input_item.image, (self.size[0], int(self.size[0] * input_item.image.shape[0] / input_item.image.shape[1])))
        elif self.size[1] is not None:
            input_item.image = cv2.resize(input_item.image, (int(self.size[1] * input_item.image.shape[1] / input_item.image.shape[0]), self.size[1]))

        return input_item