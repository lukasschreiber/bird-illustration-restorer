from pipeline import Pipeline, LoadImageStep, ResizeStep, GrayscaleStep
import cv2

# Create a new pipeline
pipeline = Pipeline()

pipeline.load_from_config("config.yaml")

output_images = pipeline.run()

if not isinstance(output_images, list):
    output_images = [output_images]
    
preview_images = pipeline.get_previews()

if not isinstance(preview_images, list):
    preview_images = [preview_images]

for entry in preview_images:
    for name, container in entry.items():
        if not isinstance(container, list):
            container = [container]
            
        for container in container:
            cv2.imshow(f"{container.title} {container.english_label} [{container.scientific_label}] - {container.page}.v.{container.instance} (physical: {container.physical_page})", container.image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()