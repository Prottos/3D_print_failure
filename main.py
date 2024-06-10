import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2

# Mapowanie klas
class_names = ['over_extrusion', 'spaghetti', 'stringing', 'under_extrusion', 'warping']
class_to_index = {name: idx for idx, name in enumerate(class_names)}

def load_image_and_labels(image_folder, labels_folder):
    images = []
    labels = []
    bboxes = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(labels_folder, filename.replace('.jpg', '.txt'))

            # Load image
            image = load_img(image_path, target_size=(640, 640))
            image = img_to_array(image)
            images.append(image)

            # Load labels
            with open(label_path, 'r') as file:
                lines = file.readlines()
                data = []

                for line in lines:
                    values = list(map(float, line.strip().split()))
                    data.extend(values)

                # Process data into the desired format: class xmin ymin xmax ymax
                formatted_data = []
                i = 0
                while i < len(data):
                    if i + 5 <= len(data):  # Check if there are enough values to unpack
                        cls = int(data[i])
                        xmin, ymin, xmax, ymax = data[i + 1:i + 5]
                        formatted_data.append([cls, xmin, ymin, xmax, ymax])
                        i += 5
                    else:
                        print(f"Warning: Incomplete data in file {label_path}, skipping remaining data.")
                        break

                for entry in formatted_data:
                    label = entry[0]
                    bbox = np.array(entry[1:5])
                    labels.append(label)
                    bboxes.append(bbox)
    return np.array(images), np.array(labels), np.array(bboxes)

# Przykład użycia
image_folder = "images"
labels_folder = "labels"
images, labels, bboxes = load_image_and_labels(image_folder, labels_folder)

# Normalize images
images = images / 255.0

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)

# Number of classes
num_classes = len(class_names)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(images.shape[1], images.shape[2], images.shape[3])),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=4, validation_data=(X_val, y_val))

# Capture video from URL
video = cv2.VideoCapture('http://192.168.8.6:5001/mjpeg')

# Function to preprocess frames
def preprocess_frame(frame):
    # Resize (while maintaining the aspect ratio) to the input size of the model
    frame = cv2.resize(frame, (images.shape[1], images.shape[2]))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Main loop; infers sequentially until you press "q"
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Get predictions
    predictions = model.predict(input_frame)

    # Draw rectangle around detected object
    for prediction in predictions:
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        confidence = prediction[class_index]
        if confidence > 0.5:  # Confidence threshold
            xmin, ymin, xmax, ymax = prediction[1:5]  # Extract normalized bounding box coordinates
            height, width, _ = frame.shape
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources when finished
video.release()
cv2.destroyAllWindows()