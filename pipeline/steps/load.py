from pipeline.base import PipelineStep, PipelineImageContainer
import cv2
import pandas as pd
import pymupdf
import os
import numpy as np

IMAGE_ROOT = lambda n: f"./data/in/raw/{n}_jp2"
PDF_PATH = lambda n: f"./data/in/raw/{n}.pdf"
INDEX_PATH = lambda n: f"./data/in/raw/{n}.csv"


class LoadImageStep(PipelineStep):
    def __init__(
        self,
        name: str,
        book: str,
        pages: list[tuple[int, int]] | list[int] | tuple[int, int] | int | None = None,
        pipeline=None,
    ):
        """
        Initialize the step with a list of (page_number, instance_number) tuples.
        """
        super().__init__(name, pipeline)
        self.book = book
        self.image_root = IMAGE_ROOT(book)

        # Load metadata
        index_path = INDEX_PATH(book)
        pdf_path = PDF_PATH(book)
        self.index = pd.read_csv(index_path)
        self.pdf = pymupdf.open(pdf_path)

        self.pipeline.set_property("pdf", self.pdf)
        self.pipeline.set_property("index", self.index)

        if isinstance(pages, int):
            self.pages = [(pages, 0)]
        elif isinstance(pages, tuple):
            self.pages = [pages]
        elif isinstance(pages, list):
            self.pages = [
                (page, 0) if isinstance(page, int) else page for page in pages
            ]

        # Create page map
        self.page_map = {}
        for n in range(len(self.pdf)):
            page = self.pdf[n]
            physical_number = n + 1
            label = page.get_label()
            if label is not None:
                self.page_map[label] = physical_number

        self.pipeline.set_property("page_map", self.page_map)

    def run(self, inputs):
        """
        Load and return a list of images for the specified pages and instances.
        """
        images = []
        for page_number, instance_number in self.pages:
            row = self.index[self.index["page"] == page_number].iloc[instance_number]
            physical_page = self.page_map[str(row["page"])]
            path = f"{self.image_root}/{self.book}_{physical_page:04d}.jp2"
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            
            images.append(
                PipelineImageContainer(
                    image=cv2.imread(path),
                    page=page_number,
                    instance=instance_number,
                    english_label=row["en_name"],
                    scientific_label=row["sci_name"],
                    physical_page=physical_page,
                )
            )
        return images
