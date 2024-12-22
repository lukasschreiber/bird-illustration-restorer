from dataclasses import dataclass
import yaml
import numpy as np

class Pipeline:
    def __init__(self):
        self.steps: list[PipelineStep] = []
        self.cache: dict[str, PipelineImageContainer] = {}
        self.global_object_storage: GlobalObjectStorage = GlobalObjectStorage()

    def add(self, step, input_name: str):
        """
        Add a step to the pipeline.
        :param step: The step to add
        :param input_names: The names of the inputs for the step
        """
        self.steps.append((step, input_name))

    def run(self):
        """
        Run the pipeline with the initial data.
        :param initial_data: The initial data to start the pipeline
        :return: The result of the pipeline
        """
        for step, input_name in self.steps:
            inputs = self.cache[input_name] if input_name else None
            result = step.run(inputs)
            self.cache[step.name] = result

        return result
    
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
            self.add(step_class(step_name, **step_args, pipeline=self), step_config.get('input', None))
    
@dataclass        
class PipelineImageContainer:
    image: np.ndarray 
    page: int
    instance: int
    english_label: str
    scientific_label: str
    physical_page: int
    
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

    def run(self, inputs: list[PipelineImageContainer] | PipelineImageContainer) -> list[PipelineImageContainer] | PipelineImageContainer:
        """
        Process a list of inputs or a single input.
        """
        if isinstance(inputs, list):
            return [self.process_single(item) for item in inputs]
        else:
            return self.process_single(inputs)