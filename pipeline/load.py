from pipeline.utils import PipelineImageContainer
import cv2
import pandas as pd
import pymupdf
import os

IMAGE_ROOT = lambda d, n: f"{d}{n}_jp2"
PDF_PATH = lambda d, n: f"{d}{n}.pdf"
INDEX_PATH = lambda d, n: f"{d}{n}.csv"

def get_pipeline_input(
    directory: str,
    book: str,
    pages: list[tuple[int, int]] | list[int] | tuple[int, int] | int | None = None,
) -> tuple[pd.DataFrame, pymupdf.Document, dict, list[PipelineImageContainer]]:
    """
    Load metadata, PDF, and images for the specified book and pages.
    
    :param book: The name of the book
    :param pages: The pages and instances to load
    :return: A tuple containing the metadata, PDF, page map, and images
    """
    image_root = IMAGE_ROOT(directory, book)
    index_path = INDEX_PATH(directory, book)
    pdf_path = PDF_PATH(directory, book)
    
    index = pd.read_csv(index_path)
    pdf = pymupdf.open(pdf_path)

    if isinstance(pages, int):
        pages = [(pages, 0)]
    elif isinstance(pages, tuple):
        pages = [pages]
    elif isinstance(pages, list):
        pages = [
            (page, 0) if isinstance(page, int) else page for page in pages
        ]
    elif pages is None:
        pages = [(page, 0) for page in index["page"].unique()]

    # Create page map
    page_map = {}
    for n in range(len(pdf)):
        page = pdf[n]
        physical_number = n + 1
        label = page.get_label()
        if label is not None:
            page_map[label] = physical_number
            
    if book == "birdsEuropeVGoul":
        page_map["448"] = 418
        
    images = []
    for page_number, instance_number in pages:
        row = index[index["page"] == page_number].iloc[instance_number]
        physical_page = page_map[str(row["page"])]
        path = f"{image_root}/{book}_{physical_page:04d}.jp2"
        
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
                book=book,
            )
        )
    return index, pdf, page_map, images

