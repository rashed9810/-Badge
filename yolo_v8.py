import torch
from PIL import Image
#Load the model function
def load_model(model_name='yolov8'):
    """
    Load the YOLOv8 model.

    Parameters:
    - model_name (str): The name of the YOLO model to load.

    Returns:
    - model: The loaded YOLO model.
    """
    model = torch.hub.load('ultralytics/yolov8', model_name)
    return model
# Prepare Image Function
def prepare_image(image_path):
    """
    Prepare the image for detection.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - img: The loaded and prepared image.
    """
    img = Image.open(image_path)
    # Insert any required transformations here
    return img


# Perform Object Detection Function

def perform_detection(model, img):
    """
    Perform object detection on the image using the specified model.

    Parameters:
    - model: The YOLO model to use for detection.
    - img: The image to detect objects in.

    Returns:
    - results: The detection results.
    """
    results = model(img)
    return results
# Display Detected Objects Function
def display_results(results):
    """
    Display the detected objects in the image.

    Parameters:
    - results: The detection results to display.
    """
    results.show()

# finally i will working on main function 
  def main():
    """
    Main function to perform object detection.
    """
    model = load_model()
    img = prepare_image("path/to/your/image.jpg")
    results = perform_detection(model, img)
    display_results(results)

if __name__ == "__main__":
    main()
