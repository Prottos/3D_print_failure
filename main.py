import torch
import cv2
import requests
import numpy as np
from roboflow import Roboflow

# rf = Roboflow(api_key="yg47kdFGg2UqYNjsOY6Z")
# project = rf.workspace("failuredetection").project("failure-detection-eitbv")
# version = project.version(1)
# dataset = version.download("yolov5")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='Failure-Detection-1/weights.pt')

conf_threshold = 0.19
model.conf = 0.19

def get_frame_from_camera(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return img
    else:
        print("Niepoprawny status {}".format(response.status_code))
    return None

def detect_objects(frame):
    results = model(frame)
    return results

def draw_boxes(frame, results):
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf >= model.conf:
            label = model.names[int(cls)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame

camera_url = 'http://192.168.8.6:5001/mjpeg'

while True:
    frame = get_frame_from_camera(camera_url)
    if frame is not None:
        results = detect_objects(frame)
        frame = draw_boxes(frame, results)
        cv2.imshow('Detekcja błędów', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Błąd wczytywania obrazu z kamery")

cv2.destroyAllWindows()