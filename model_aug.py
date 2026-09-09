from tensorflow import keras

# 1. Load and prepare CIFAR-10 data first
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


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
    epochs=2,
    validation_split=0.1
)

test_loss, test_accuracy = model_aug.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test accuracy:", test_accuracy)

