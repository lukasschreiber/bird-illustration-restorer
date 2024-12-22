from pipeline import Pipeline, LoadImageStep, ResizeStep, GrayscaleStep
import cv2

# Create a new pipeline
pipeline = Pipeline()

pipeline.load_from_config("config.yaml")

output_images = pipeline.run()

if not isinstance(output_images, list):
    output_images = [output_images]

for i, container in enumerate(output_images):
    cv2.imshow(f"Output {container.english_label} [{container.scientific_label}] - {container.page}.{container.instance} (physical: {container.physical_page})", container.image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()