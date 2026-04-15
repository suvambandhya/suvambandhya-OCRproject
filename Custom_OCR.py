import cv2
import pytesseract
import numpy as np

def load_yolo(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

def detect_objects(image_path, net, output_layers, classes, conf_thresh=0.5, nms_thresh=0.4):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_thresh:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    cropped_images, detected_labels = [], []
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        crop = img[y:y+h, x:x+w]
        cropped_images.append(crop)
        detected_labels.append(label)
    return cropped_images, detected_labels

def ocr_images(cropped_images):
    texts = []
    for img in cropped_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 6')
        texts.append(text.strip())
    return texts
