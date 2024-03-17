import cv2
import torch

class YOLOv5:
    def __init__(self, model_name='yolov5s.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)

    def detect_objects(self, image):
        return self.model(image)

def load_image(image_path):
    return cv2.imread(image_path)

def draw_boxes(image, results):
    labels, coordinates = results.xyxyn[0][:, -1].detach().cpu().numpy(), results.xyxyn[0][:, :-1].detach().cpu().numpy()
    for label, (x, y, w, h) in zip(labels, coordinates):
        x1, y1, x2, y2 = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'{model.names[int(label)]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

def display_image(image):
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Replace with your image path
    image_path = 'path/to/your/image.jpg'
    
    # Initialize YOLOv5 model
    yolov5 = YOLOv5()

    # Load image
    image = load_image(image_path)

    # Perform object detection
    results = yolov5.detect_objects(image)

    # Draw bounding boxes and labels
    image_with_boxes = draw_boxes(image, results)

    # Display the image
    display_image(image_with_boxes)
