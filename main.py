import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Funkcja do załadowania obrazów
def load_images(image_folder):
    images = []
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        images.append(image)
    return images, image_files

<<<<<<< HEAD
conf_threshold = 0.15
model.conf = 0.15

camera_url = 'http://192.168.8.6:5001/mjpeg'

def get_frame_from_camera(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=4096):
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
=======

# Funkcja do załadowania etykiet
def load_labels(label_folder, image_files, image_size):
    labels = []
    for image_file in image_files:
        label_file = image_file.replace('.jpg', '.txt')  # Assuming the images are in .jpg format
        label_path = os.path.join(label_folder, label_file)
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for image {image_file}")
            labels.append((None, None))
            continue
        with open(label_path, 'r') as file:
            content = file.read()
            label_lines = content.strip().split('\n')
            for line in label_lines:
                values = [float(i) for i in line.split()]
                if len(values[1:]) % 4 != 0:  # Sprawdzanie poprawności długości etykiet
                    print(f"Warning: Invalid label format in file {label_file}, line: {line}")
                    labels.append((None, None))
                    break
                class_id = values[0]
                bbox = np.array(values[1:]).reshape(-1, 4)
                bbox[:, [0, 2]] *= image_size[0]  # Przeskalowanie xmin, xmax do szerokości obrazu
                bbox[:, [1, 3]] *= image_size[1]  # Przeskalowanie ymin, ymax do wysokości obrazu
                labels.append((class_id, bbox))
    return labels
>>>>>>> parent of a9d1c47 (Zmiana koncepcji)


# Tworzenie modelu
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(640, 640, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

<<<<<<< HEAD
while True:
    frame = get_frame_from_camera(camera_url)
    if frame is not None:
        results = detect_objects(frame)
        frame = draw_boxes(frame, results)
        cv2.imshow('Detekcja błędów', frame)
=======

# Ładowanie danych
image_folder = 'images'
label_folder = 'labels'
image_size = (640, 640)

images, image_files = load_images(image_folder)
labels = load_labels(label_folder, image_files, image_size)

# Filtracja obrazów i etykiet, aby usunąć błędne wpisy
valid_images = []
valid_labels = []
for img, lbl in zip(images, labels):
    if lbl[0] is not None:
        valid_images.append(img)
        valid_labels.append(lbl[0])

# Przygotowanie danych do treningu
images_resized = np.array([cv2.resize(image, image_size) for image in valid_images])
labels_resized = np.array(valid_labels)  # Przyjmujemy, że pierwsza etykieta jest główną klasą

# Sprawdzenie, czy liczba obrazów i etykiet jest taka sama
assert len(images_resized) == len(labels_resized), "Liczba obrazów i etykiet nie jest zgodna!"

# Trening modelu
model = create_model()
model.fit(images_resized, labels_resized, epochs=1, batch_size=32)


# Funkcja do przechwytywania wideo i predykcji
def video_capture_and_predict(model, video_url):
    cap = cv2.VideoCapture(video_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Przetwarzanie klatki
        frame_resized = cv2.resize(frame, (640, 640))
        frame_expanded = np.expand_dims(frame_resized, axis=0)

        # Predykcja
        predictions = model.predict(frame_expanded)
        predicted_class = np.argmax(predictions[0])

        # Wyświetlanie wyniku na obrazie
        class_names = ['over_extrusion', 'spaghetti', 'stringing', 'under_extrusion', 'warping']
        cv2.putText(frame, class_names[predicted_class], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Wyświetlanie klatki
        cv2.imshow('Video', frame)

        # Zatrzymanie na 'q'
>>>>>>> parent of a9d1c47 (Zmiana koncepcji)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


video_url = "http://192.168.8.6:5001/mjpeg"
video_capture_and_predict(model, video_url)