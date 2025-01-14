import yaml
import numpy as np
from dataclasses import replace
import cv2
from .load import get_pipeline_input
from .save import save_pipeline_output
from typing import Callable
from .utils import PipelineImageContainer, PreviewImage, GlobalObjectStorage

class Pipeline:
    def __init__(self):
        self.steps: list[tuple[PipelineStep, str, str]] = []
        self.cache: dict[str, PipelineImageContainer] = {}
        self.preview_enabled: bool = False
        self.preview_images: dict[str, bool] = {}
        self.preview_image_titles: dict[str, str] = {}
        self.global_object_storage: GlobalObjectStorage = GlobalObjectStorage()
        self.input_directory: str = ""
        self.book: str = ""
        self.preview_size: int = 800
        self.pages: list[tuple[int, int]] | list[int] | tuple[int, int] | int | None = None
        self.output_directory: str = ""

    def add(self, step, input_name: str, mask_name: str = None) -> None:
        """
        Add a step to the pipeline.
        :param step: The step to add
        :param input_names: The names of the inputs for the step
        """
        self.steps.append((step, input_name, mask_name))
        
    def run_all(self, preview_callback: Callable[[PreviewImage], None]) -> list[PipelineImageContainer]:
        [index, pdf, page_map, images] = get_pipeline_input(self.input_directory, self.book, self.pages)
        results = []
        
        self.set_property("index", index)
        self.set_property("pdf", pdf)
        self.set_property("page_map", page_map)
        
        from .steps import ResizeStep

        for image in images:
            for index, [step, input_name, mask_name] in enumerate(self.steps):
                if index == 0:
                    input = image
                else:
                    input = self.cache[self.get_cache_key(image, input_name)] if input_name else None
                mask = self.cache[self.get_cache_key(image, mask_name)] if mask_name else None
                result = step.run(input, mask)
                self.cache[self.get_cache_key(image, step.name)] = result
                                        
                if self.preview_enabled and step.name in self.preview_images:
                    preview = PreviewImage(
                        image=ResizeStep(None, max=self.preview_size, pipeline=self).process_single(replace(result)).image,
                        page=result.page,
                        instance=result.instance,
                        english_label=result.english_label,
                        scientific_label=result.scientific_label,
                        physical_page=result.physical_page,
                        title=self.preview_image_titles[step.name]
                    )
                    preview_callback(preview)         
            results.append(result)
            save_pipeline_output(self.output_directory, result)
            
        return results
    
    def get_cache_key(self, image: PipelineImageContainer, name: str) -> str:
        return f"{image.page}_{image.instance}_{name}"
    
    def get_property(self, name: str, type_hint: type = None) -> any:
        """
        Get a property from the global object storage.
        """
        value = self.global_object_storage.get(name)
        if type_hint and not isinstance(value, type_hint):
            raise TypeError(f"Expected type {type_hint}, but got {type(value)}")
        return value
    
    def set_property(self, name: str, value: any):
        """
        Set a property in the global object storage.
        """
        if name in self.global_object_storage.storage:
            raise ValueError(f"Property '{name}' already exists in global storage")
        self.global_object_storage.set(name, value)

    def load_from_config(self, config_path: str) -> None:
        """
        Load the pipeline configuration from a YAML file.
        :param config_path: The path to the configuration file
        """
        from .steps import STEP_REGISTRY

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for step_config in config['pipeline']['steps']:
            step_name = step_config['output']
            step_class = STEP_REGISTRY[step_config['step']]
            step_args = step_config.get('parameters', {})
            self.add(step_class(step_name, **step_args, pipeline=self), step_config.get('input', None), step_config.get('mask', None))
            
        self.input_directory = config['pipeline']['input']['directory']
        self.book = config['pipeline']['input']['book']
        self.output_directory = config['pipeline']['output']['directory']
        self.pages = config['pipeline']['input']['pages'] if 'pages' in config['pipeline']['input'] else None
            
        if 'preview' in config:
            self.preview_enabled = config['preview']['enabled'] if 'enabled' in config['preview'] else False
            self.preview_size = config['preview']['size'] if 'size' in config['preview'] else 800
            if self.preview_enabled:
                for preview_config in config['preview']['images']:
                    if 'enabled' in preview_config and not preview_config['enabled']:
                        continue
                    self.preview_images[preview_config['name']] = True
                    self.preview_image_titles[preview_config['name']] = preview_config.get('title', preview_config['name'])
 
            
class PipelineStep:
    def __init__(self, name: str, pipeline: Pipeline = None):
        self.name = name
        self.pipeline = pipeline

    def process_single(self, image: PipelineImageContainer) -> PipelineImageContainer:
        """
        Subclasses should implement this to process a single input.
        """
        raise NotImplementedError("Subclasses must implement 'process_single'.")

    def process_masked_single(self, image: PipelineImageContainer, mask: PipelineImageContainer | None = None) -> PipelineImageContainer:
        if mask is None:
            return self.process_single(image)
        
        if image.image.shape[:2] != mask.image.shape[:2]:
            raise ValueError("The dimensions of the mask must match the dimensions of the image.")
        
        if len(mask.image.shape) > 2 or mask.image.dtype != np.uint8:
            raise ValueError("The mask must be a single-channel binary image with dtype=np.uint8.")

        masked_image = image.image.copy()
        roi = cv2.bitwise_and(masked_image, masked_image, mask=mask.image)
        roi_container = replace(image, image=roi)
        processed_roi_container = self.process_single(roi_container)
        processed_image = masked_image.copy()
        processed_image[mask.image > 0] = processed_roi_container.image[mask.image > 0]

        return replace(image, image=processed_image)

    def run(self, input: PipelineImageContainer, mask: PipelineImageContainer | None = None) -> PipelineImageContainer:
        """
        Process a list of inputs or a single input.
        """
        mask = mask if isinstance(mask, PipelineImageContainer) else None
        return self.process_masked_single(replace(input), mask)