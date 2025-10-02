import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Directories for training and validation
train_dir = r"C:\Users\sanga\Downloads\Image_Classification\train"
val_dir = r"C:\Users\sanga\Downloads\Image_Classification\validation"

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Define the training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True  # Enable shuffling for better training
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Print dataset sizes
print("Training samples:", train_generator.samples)
print("Validation samples:", validation_generator.samples)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Helps reduce overfitting
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for better stability
    metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss", mode="min")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
    epochs=80,  # Increase the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
    callbacks=[checkpoint, early_stop]
)

# Evaluate the model
score = model.evaluate(validation_generator, steps=max(1, validation_generator.samples // validation_generator.batch_size))
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# Save the trained model
model.save("skin_lesion_classifier.keras")  # Save the model in recommended format
print("Model saved successfully!")