import numpy as np
from dataclasses import dataclass

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