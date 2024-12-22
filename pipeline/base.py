import yaml

class Pipeline:
    def __init__(self):
        self.steps = []
        self.cache = {}

    def add(self, step, input_name):
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
            self.add(step_class(step_name, **step_args), step_config.get('input', None))
            
            
            
class PipelineStep:
    def __init__(self, name):
        self.name = name

    def process_single(self, input_item):
        """
        Subclasses should implement this to process a single input.
        """
        raise NotImplementedError("Subclasses must implement 'process_single'.")

    def run(self, inputs):
        """
        Process a list of inputs or a single input.
        """
        if isinstance(inputs, list):
            return [self.process_single(item) for item in inputs]
        else:
            return self.process_single(inputs)