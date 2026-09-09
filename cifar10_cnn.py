from tensorflow import keras
import matplotlib.pyplot as plt



# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)

plt.imshow(X_train[0])
plt.axis("off")
plt.show()

print("Label:", y_train[0])

model = keras.Sequential([
    keras.layers.Input(shape=[32, 32, 3]),

    keras.layers.Conv2D(
        32,
        kernel_size=3,
        activation="relu"
    ),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(
        64,
        kernel_size=3,
        activation="relu"
    ),

    keras.layers.MaxPooling2D(pool_size=2),

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

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=1,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

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

image = X_test[0]

print("Actual class:", class_names[int(y_test[0])])

prediction = model.predict(
    image.reshape(1, 32, 32, 3),
    verbose=0
)

predicted_class = prediction.argmax()

print("Predicted class:", class_names[predicted_class])

import matplotlib.pyplot as plt

plt.imshow(image)
plt.title(
    f"Predicted: {class_names[predicted_class]} | "
    f"Actual: {class_names[int(y_test[0])]}"
)
plt.axis("off")
plt.show()

