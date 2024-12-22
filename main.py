from pipeline import Pipeline, LoadImageStep, ResizeStep, GrayscaleStep
import cv2

# Create a new pipeline
pipeline = Pipeline()

pipeline.load_from_config("config.yaml")

output_images = pipeline.run()
for i, container in enumerate(output_images):
    cv2.imshow(f"Image {i}", container.image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()