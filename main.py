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


# Tworzenie modelu
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(640, 640, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(5 + 4, activation='sigmoid')  # 5 klas + 4 wartości bbox (xmin, ymin, xmax, ymax)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


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
        valid_labels.append(lbl)

# Przygotowanie danych do treningu
images_resized = np.array([cv2.resize(image, image_size) for image in valid_images])
labels_resized = []
for lbl in valid_labels:
    class_id, bbox = lbl
    label_array = np.zeros(5 + 4)  # 5 klas + 4 wartości bbox
    label_array[int(class_id)] = 1  # One-hot encoding dla klasy
    label_array[5:] = bbox.flatten()  # Dodanie bbox
    labels_resized.append(label_array)
labels_resized = np.array(labels_resized)

# Sprawdzenie, czy liczba obrazów i etykiet jest taka sama
assert len(images_resized) == len(labels_resized), "Liczba obrazów i etykiet nie jest zgodna!"

# Trening modelu
model = create_model()
model.fit(images_resized, labels_resized, epochs=1, batch_size=32)


# Funkcja do rysowania prostokątów na klatkach wideo
def draw_bounding_boxes(frame, bboxes, class_id):
    class_names = ['over_extrusion', 'spaghetti', 'stringing', 'under_extrusion', 'warping']
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, class_names[int(class_id)], (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)


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
        predicted_class = np.argmax(predictions[0][:5])  # Przewidywanie klasy
        predicted_bboxes = predictions[0][5:].reshape(-1, 4)  # Przewidywanie bbox

        # Rysowanie prostokątów na klatce
        draw_bounding_boxes(frame, predicted_bboxes, predicted_class)

        # Wyświetlanie klatki
        cv2.imshow('Video', frame)

        # Zatrzymanie na 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


video_url = "http://192.168.8.6:5001/mjpeg"
video_capture_and_predict(model, video_url)