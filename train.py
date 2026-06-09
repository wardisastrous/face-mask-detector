import os
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

dataset_path = "dataset"

data = []
labels = []

# Load images
for category in ["with_mask", "without_mask"]:
    path = os.path.join(dataset_path, category)
    label = 1 if category == "with_mask" else 0

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# Train-test split
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Convert labels
lb = LabelBinarizer()
trainY = to_categorical(lb.fit_transform(trainY))
testY = to_categorical(lb.transform(testY))

# Load MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=(224, 224, 3))

# Freeze layers
for layer in baseModel.layers:
    layer.trainable = False

# Add head
head = baseModel.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

model = Model(inputs=baseModel.input, outputs=head)

# Compile
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=1e-4),
              metrics=["accuracy"])

# Train
H = model.fit(trainX, trainY,
              validation_data=(testX, testY),
              epochs=10,
              batch_size=32)

# Save model
model.save("mask_detector.keras")

# Plot accuracy
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Accuracy")
plt.show()