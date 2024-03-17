import torch
from PIL import Image

# Load the model
model = torch.hub.load('ultralytics/yolov8', 'yolov8')

# Prepare your image
image_path = "path/to/your/image.jpg"
img = Image.open(image_path)

# Perform detection
results = model(img)

# Show results
results.show()
