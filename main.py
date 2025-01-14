from pipeline import Pipeline, PreviewImage
import cv2

# Create a new pipeline
pipeline = Pipeline()

pipeline.load_from_config("config.yaml")

def show_preview(image: PreviewImage):
    cv2.imshow(f"{image.title} {image.english_label} [{image.scientific_label}] - {image.page}.v.{image.instance} (physical: {image.physical_page})", image.image)
    cv2.waitKey(1)
output_images = pipeline.run_all(show_preview)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()