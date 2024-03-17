import cv2
import torch
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

def detect_objects(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Perform object detection
    results = model(img)

    # Extract detection results
    labels, coordinates = results.xyxyn[0][:, -1].detach().cpu().numpy(), results.xyxyn[0][:, :-1].detach().cpu().numpy()

    # Draw bounding boxes and labels
    for label, (x, y, w, h) in zip(labels, coordinates):
        x1, y1, x2, y2 = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'{model.names[int(label)]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image
    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Replace with your image path
    image_path = 'path/to/your/image.jpg'
    detect_objects(image_path)
