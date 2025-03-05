import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Carregue as labels
labels = np.load('labels.npy', allow_pickle=True)
num_classes = len(labels)

# Defina o modelo
model = Sequential()
model.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', 
                 input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=4))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# Compile o modelo
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Carregue os geradores de dados
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(50, 50),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(50, 50),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Callback para salvar o melhor modelo durante o treinamento
checkpoint = ModelCheckpoint('libras_model_v2.keras', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max')

# Treine o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    callbacks=[checkpoint])

print("Treinamento concluído! Modelo salvo como 'libras_model_v2.keras'.")