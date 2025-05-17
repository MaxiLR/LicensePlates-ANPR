from fast_plate_ocr import ONNXPlateRecognizer
from inference import get_model
import supervision as sv
import numpy as np
import dotenv
import cv2

m = ONNXPlateRecognizer("global-plates-mobile-vit-v2-model")
model = get_model(model_id="license-plates-xfeyr/1")
dotenv.load_dotenv()


def detect_plates(image) -> list:
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    return detections


def read_plates(image, detections):
    labels = {}
    for i in range(len(detections)):
        x1, y1, x2, y2 = map(int, detections.xyxy[i])

        plate_image = image[y1:y2, x1:x2]
        plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        plate_text = m.run(plate_image_gray)
        plate_text[0] = plate_text[0].replace("_", "")

        labels[plate_text[0]] = (x1, y1, x2, y2)

    return labels


def annotate(image, labels):
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for label, (x1, y1, x2, y2) in labels.items():
        xyxy = np.array([[x1, y1, x2, y2]])
        class_id = np.array([0])

        detections = sv.Detections(xyxy=xyxy, class_id=class_id)

        image = bounding_box_annotator.annotate(scene=image, detections=detections)
        image = label_annotator.annotate(
            scene=image, detections=detections, labels=[label]
        )

    return image


if __name__ == "__main__":
    image_file = "test.jpg"
    image = cv2.imread(image_file)
    detections = detect_plates(image)
    labels = read_plates(image, detections)
    annotated_image = annotate(image, labels)
    sv.plot_image(annotated_image)
