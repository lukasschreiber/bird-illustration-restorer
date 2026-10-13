"""Generate a repeatable cross-volume visual review without replacing originals."""
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from pipeline import Pipeline
from pipeline.load import get_pipeline_input
from pipeline.save import output_filename


SAMPLES = {
    "birdsEuropeIGoul": [1, 10, 34],
    "birdsEuropeIIGoul": [100, 110],
    "birdsEuropeIIIGoul": [150, 160],
    "birdsEuropeIVGoul": [243, 278],
    "birdsEuropeVGoul": [354, 355, 356],
}


def thumbnail(image):
    canvas = np.full((270, 440, 3), 255, np.uint8)
    h, w = image.shape[:2]
    scale = min(420 / w, 235 / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)))
    canvas[30:30 + resized.shape[0], 10:10 + resized.shape[1]] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/out/segmentation-review")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows, metrics = [], []
    for book, pages in SAMPLES.items():
        pipeline = Pipeline()
        pipeline.load_from_config(args.config)
        _, pdf, page_map, images = get_pipeline_input(pipeline.input_directory, book, pages)
        pipeline.set_property("pdf", pdf)
        pipeline.set_property("page_map", page_map)
        try:
            for image in images:
                cv2.setRNGSeed(0)
                for step, source, mask_name in pipeline.steps:
                    item = image if source == "initial_image" else pipeline.cache[pipeline.get_cache_key(image, source)]
                    mask = pipeline.cache[pipeline.get_cache_key(image, mask_name)] if mask_name else None
                    result = step.run(item, mask)
                    pipeline.cache[pipeline.get_cache_key(image, step.name)] = result
                mask = pipeline.cache[pipeline.get_cache_key(image, "subject_mask")].image
                assert result.image.size and np.all(np.isin(mask, [0, 255]))
                cv2.imwrite(str(root / f"{book}-{image.page}.jpg"), result.image)
                previous_path = Path('data/out/processed') / book / output_filename(image)
                if not previous_path.exists():
                    previous_path = Path('data/out/processed') / book / f'{image.page}.jpg'
                previous = cv2.imread(str(previous_path))
                row = np.hstack([thumbnail(previous if previous is not None else image.image), thumbnail(result.image)])
                cv2.putText(row, f"{book} {image.page}: previous / source mask", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
                rows.append(row)
                metrics.append({"book": book, "page": image.page,
                                "coverage": round(float(np.mean(mask > 0)), 4),
                                "size": result.image.shape[:2]})
                pipeline.cache.clear()
                print(metrics[-1], flush=True)
        finally:
            pdf.close()
    cv2.imwrite(str(root / "comparison.jpg"), np.vstack(rows))
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
