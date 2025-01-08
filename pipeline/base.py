from dataclasses import dataclass
import yaml
import numpy as np
from dataclasses import replace
import cv2

@dataclass        
class PipelineImageContainer:
    image: np.ndarray 
    page: int
    instance: int
    english_label: str
    scientific_label: str
    physical_page: int
    book: str
    
@dataclass
class PreviewImage:
    image: np.ndarray | None
    page: int | None
    instance: int | None
    english_label: str | None
    scientific_label: str | None
    physical_page: int | None
    title: str

class Pipeline:
    def __init__(self):
        self.steps: list[tuple[PipelineStep, str, str]] = []
        self.cache: dict[str, PipelineImageContainer] = {}
        self.preview_enabled: bool = False
        self.preview_images: dict[str, PreviewImage] = {}
        self.preview_image_titles: dict[str, str] = {}
        self.global_object_storage: GlobalObjectStorage = GlobalObjectStorage()

    def add(self, step, input_name: str, mask_name: str = None) -> None:
        """
        Add a step to the pipeline.
        :param step: The step to add
        :param input_names: The names of the inputs for the step
        """
        self.steps.append((step, input_name, mask_name))

    def run(self) -> list[PipelineImageContainer] | PipelineImageContainer:
        """
        Run the pipeline with the initial data.
        :param initial_data: The initial data to start the pipeline
        :return: The result of the pipeline
        """
        for step, input_name, mask_name in self.steps:
            inputs = self.cache[input_name] if input_name else None
            masks = self.cache[mask_name] if mask_name else None
            result = step.run(inputs, masks)
            self.cache[step.name] = result
                                    
            if self.preview_enabled and step.name in self.preview_images:
                preview_result = [result] if not isinstance(result, list) else result
                preview_images = []
                for preview_result_item in preview_result:
                    preview_image = PreviewImage(image=preview_result_item.image, page=preview_result_item.page, instance=preview_result_item.instance, english_label=preview_result_item.english_label, scientific_label=preview_result_item.scientific_label, physical_page=preview_result_item.physical_page, title=self.preview_image_titles[step.name])
                    preview_images.append(preview_image)
                    
                self.preview_images[step.name] = preview_images if len(preview_images) > 1 else preview_images[0]                

        return result
    
    def get_previews(self, size: int = 800) -> dict[str, PreviewImage | list[PreviewImage]]:
        """
        Get the preview images.
        """
        for preview_image in self.preview_images.values():
            from .steps import ResizeStep
            if isinstance(preview_image, list):
                for img in preview_image:
                    img.image = ResizeStep(None, max=size, pipeline=self).process_single(replace(img)).image
            else:
                preview_image.image = ResizeStep(None, max=size, pipeline=self).process_single(replace(preview_image)).image
            # print([img.title for img in preview_image] if isinstance(preview_image, list) else preview_image.title)
        return self.preview_images
    
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
            
        for step_config in config['pipeline']:
            step_name = step_config['output']
            step_class = STEP_REGISTRY[step_config['step']]
            step_args = step_config.get('parameters', {})
            self.add(step_class(step_name, **step_args, pipeline=self), step_config.get('input', None), step_config.get('mask', None))
            
        if 'preview' in config:
            self.preview_enabled = config['preview']['enabled'] if 'enabled' in config['preview'] else False
            if self.preview_enabled:
                for preview_config in config['preview']['images']:
                    if 'enabled' in preview_config and not preview_config['enabled']:
                        continue
                    self.preview_images[preview_config['name']] = None
                    self.preview_image_titles[preview_config['name']] = preview_config.get('title', preview_config['name'])

    
class GlobalObjectStorage:
    def __init__(self):
        self.storage: dict[str, any] = {}

    def get(self, name: str) -> any:
        return self.storage.get(name, None)

    def set(self, name: str, value: any) -> None:
        self.storage[name] = value

    def remove(self, name: str) -> None:
        if name in self.storage:
            del self.storage[name]

    def exists(self, name: str) -> bool:
        return name in self.storage       
            
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

    def run(self, inputs: list[PipelineImageContainer] | PipelineImageContainer, masks: list[PipelineImageContainer] | PipelineImageContainer | None = None) -> list[PipelineImageContainer] | PipelineImageContainer:
        """
        Process a list of inputs or a single input.
        """
        if isinstance(inputs, list):
            if masks and not isinstance(masks, list):
                raise ValueError("Masks should be a list if inputs are a list.")
            return [
                self.process_masked_single(replace(item), mask=(masks[i] if masks else None)) 
                for i, item in enumerate(inputs)
            ]
        else:
            mask = masks if isinstance(masks, PipelineImageContainer) else None
            return self.process_masked_single(replace(inputs), mask)