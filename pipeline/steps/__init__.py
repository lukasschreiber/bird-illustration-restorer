import pkgutil
import inspect
from .load import LoadImageStep
from .resize import ResizeStep
from .gray import GrayscaleStep
from .fix_orientation import FixOrientationStep
from .reduce_yellow import ReduceYellowStep
from .whitebalance import WhiteBalanceStep

# Automatically register all steps in this module
STEP_REGISTRY = {}

# Loop through all files in the current directory and find classes
for _, module_name, _ in pkgutil.iter_modules(__path__):
    module = __import__(f"{__name__}.{module_name}", fromlist=["*"])
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            STEP_REGISTRY[name] = obj

# Expose all steps in the module, now all classes are available through STEP_REGISTRY
__all__ = list(STEP_REGISTRY.keys())