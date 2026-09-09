import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# PRACTICAL 45 — LOAD CIFAR-10 DATASET
# ============================================================

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
y_test = y_test.flatten()
y_train = y_train.flatten()


print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)


# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# PRACTICAL 45 — DISPLAY ONE IMAGE
# ============================================================

plt.imshow(X_train[0])

plt.title(
    f"Training Image - "
    f"{class_names[int(y_train[0])]}"
)

plt.axis("off")
plt.show()


# ============================================================
# PRACTICAL 46 — BUILD CNN
# ============================================================

model = keras.Sequential([

    keras.layers.Input(shape=[32, 32, 3]),

    keras.layers.Conv2D(
        32,
        kernel_size=3,
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=2
    ),

    keras.layers.Conv2D(
        64,
        kernel_size=3,
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=2
    ),

    keras.layers.Flatten(),

    keras.layers.Dense(
        128,
        activation="relu"
    ),

    keras.layers.Dense(
        10,
        activation="softmax"
    )
])


print("\n================ MODEL SUMMARY ================\n")

model.summary()


# ============================================================
# PRACTICAL 47 — COMPILE AND TRAIN
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n================ TRAINING ================\n")

history = model.fit(
    X_train,
    y_train,
    epochs=1,
    validation_split=0.1
)


# ============================================================
# PRACTICAL 48 — EVALUATE MODEL
# ============================================================

print("\n================ TEST RESULTS ================\n")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# ============================================================
# PRACTICAL 49 — MAKE ONE PREDICTION
# ============================================================

print("\n================ SINGLE PREDICTION ================\n")

image = X_test[0]

prediction = model.predict(
    image.reshape(1, 32, 32, 3),
    verbose=0
)

predicted_class = prediction.argmax()

print(
    "Predicted:",
    class_names[predicted_class]
)

print(
    "Actual:",
    class_names[int(y_test[0])]
)


# Display image

plt.imshow(image)

plt.title(
    f"Predicted: {class_names[predicted_class]}\n"
    f"Actual: {class_names[int(y_test[0].item())]}"
)

plt.axis("off")
plt.show()


# ============================================================
# PRACTICAL 50 — MULTIPLE PREDICTIONS
# ============================================================

print("\n================ MULTIPLE PREDICTIONS ================\n")

for i in range(10):

    image = X_test[i]

    prediction = model.predict(
        image.reshape(1, 32, 32, 3),
        verbose=0
    )

    predicted_class = prediction.argmax()

    actual_class = int(y_test[i])

    print(
        f"Image {i}: "
        f"Predicted = {class_names[predicted_class]}, "
        f"Actual = {class_names[actual_class]}"
    )


# ============================================================
# PRACTICAL 50 — VISUALIZE MULTIPLE PREDICTIONS
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(10):

    image = X_test[i]

    prediction = model.predict(
        image.reshape(1, 32, 32, 3),
        verbose=0
    )

    predicted_class = prediction.argmax()

    actual_class = int(y_test[i])

    plt.subplot(2, 5, i + 1)

    plt.imshow(image)

    plt.title(
        f"P: {class_names[predicted_class]}\n"
        f"A: {class_names[actual_class]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# PRACTICAL 51 — FIND FIRST INCORRECT PREDICTION
# ============================================================

print(
    "\n================ FIRST INCORRECT "
    "PREDICTION ================\n"
)

for i in range(len(X_test)):

    image = X_test[i]

    prediction = model.predict(
        image.reshape(1, 32, 32, 3),
        verbose=0
    )

    predicted_class = prediction.argmax()

    actual_class = int(y_test[i])

    if predicted_class != actual_class:

        print("Image index:", i)

        print(
            "Predicted:",
            class_names[predicted_class]
        )

        print(
            "Actual:",
            class_names[actual_class]
        )

        break


# Display incorrect image

plt.imshow(X_test[i])

plt.title(
    f"Predicted: {class_names[predicted_class]}\n"
    f"Actual: {class_names[actual_class]}"
)

plt.axis("off")
plt.show()


# ============================================================
# PRACTICAL 52 — PREDICTION CONFIDENCE
# ============================================================

print(
    "\n================ PREDICTION "
    "CONFIDENCE ================\n"
)

image = X_test[0]

prediction = model.predict(
    image.reshape(1, 32, 32, 3),
    verbose=0
)[0]


print("\nClass probabilities:\n")

for i, probability in enumerate(prediction):

    print(
        f"{class_names[i]}: "
        f"{probability:.4f}"
    )


predicted_class = prediction.argmax()

confidence = prediction[predicted_class]

print(
    "\nPredicted class:",
    class_names[predicted_class]
)

print(
    "Confidence:",
    confidence
)


# ============================================================
# PRACTICAL 53 — CONFUSION MATRIX
# ============================================================

print(
    "\n================ CONFUSION MATRIX ================\n"
)

# Generate predictions for entire test dataset

y_pred = model.predict(
    X_test,
    verbose=0
)

# Convert probabilities into class numbers

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

# Convert actual labels from shape (10000, 1)
# to shape (10000,)

y_true = y_test.flatten()


# Create confusion matrix

cm = confusion_matrix(
    y_true,
    y_pred_classes
)

print(cm)


# Visualize confusion matrix

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    np.arange(10),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(10),
    class_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("CIFAR-10 Confusion Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# PRACTICAL 54 — PRECISION, RECALL AND F1-SCORE
# ============================================================

print(
    "\n================ CLASSIFICATION REPORT ================\n"
)

report = classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names
)

print(report)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n========================================================"
)

print(
    "CIFAR-10 CNN PROJECT COMPLETED"
)

print(
    "========================================================"
)

model.save("cifar10_cnn.keras")

print("Model saved successfully.")

loaded_model = keras.models.load_model(
    "cifar10_cnn.keras"
)

print("Model loaded successfully.")

test_loss, test_accuracy = loaded_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test accuracy:", test_accuracy)

image = X_test[0]

prediction = loaded_model.predict(
    image.reshape(1, 32, 32, 3),
    verbose=0
)

predicted_class = prediction.argmax()

print(
    "Predicted:",
    class_names[predicted_class]
)

print(
    "Actual:",
    class_names[int(y_test[0])]
)

test_loss, test_accuracy = loaded_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test accuracy:", test_accuracy)

image = X_test[0]

prediction = loaded_model.predict(
    image.reshape(1, 32, 32, 3),
    verbose=0
)

predicted_class = prediction.argmax()

print(
    "Predicted:",
    class_names[predicted_class]
)

print(
    "Actual:",
    class_names[int(y_test[0])]
)

from tensorflow import keras


model_aug = keras.Sequential([

    keras.layers.Input(shape=[32, 32, 3]),

    # Data augmentation
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomTranslation(0.1, 0.1),

    # CNN
    keras.layers.Conv2D(
        32,
        kernel_size=3,
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=2
    ),

    keras.layers.Conv2D(
        64,
        kernel_size=3,
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=2
    ),

    keras.layers.Flatten(),

    keras.layers.Dense(
        128,
        activation="relu"
    ),

    keras.layers.Dense(
        10,
        activation="softmax"
    )
])


model_aug.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history_aug = model_aug.fit(
    X_train,
    y_train,
    epochs=1,
    validation_split=0.1
)

test_loss, test_accuracy = model_aug.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test accuracy:", test_accuracy)


