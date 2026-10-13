from pipeline import Pipeline, PreviewImage
import cv2
import argparse

# Create a new pipeline
pipeline = Pipeline()

parser = argparse.ArgumentParser(description="Process bird illustration plates")
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--no-preview", action="store_true")
parser.add_argument("--all", action="store_true",
                    help="Regenerate every image in the configured book without previews")
args = parser.parse_args()
pipeline.load_from_config(args.config)
if args.all:
    pipeline.pages = None
if args.no_preview or args.all:
    pipeline.preview_enabled = False

def show_preview(image: PreviewImage):
    if image.image is not None:
        cv2.imshow(f"{image.title} {image.english_label} [{image.scientific_label}] - {image.page}.v.{image.instance} (physical: {image.physical_page})", image.image)
        cv2.waitKey(1)
output_images = pipeline.run_all(show_preview)

# Wait for a key press and close the window
if pipeline.preview_enabled:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
