import keras
import numpy as np
import tensorflow as tf
from pathlib import Path

from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.python.keras.callbacks import EarlyStopping

batch_size = 32
img_height = 180
img_width = 180

# Путь к директории с данными
# Path to the directory with data
data_set = Path('./dataset')

# Тренировочная выборка
# Training dataset
train_ds = keras.utils.image_dataset_from_directory(
    data_set,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# Валидационная выборка
# Validation dataset
val_ds = keras.utils.image_dataset_from_directory(
    data_set,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
class_names = train_ds.class_names

# Метки
# Labels
labels = []
for _, label_batch in train_ds:  # Пакет данных и меток
    # Batch of data and labels
    labels.extend(label_batch.numpy())  # Добавление меток в список
    # Add labels to the list

# Подсчет количества каждого класса с помощью Counter
# Counting each class using Counter
class_counts = Counter(labels)

# Создание столбчатого графика и смотрим распределение классов в train_ds
# Creating a bar chart and viewing class distribution in train_ds
plt.figure(figsize=(10, 9))
plt.bar(class_names, [class_counts[i] for i in range(len(class_names))], color='skyblue')
plt.xlabel('Классы', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.title('Распределение классов в train_ds', fontsize=16)
plt.xticks(rotation=45)
plt.show()

# Просмотр первых 9 изображений из train_ds
# Viewing the first 9 images from train_ds
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):  # Берем один пакет изображений
    # Take one batch of images
    for i in range(9):  # Отобразим первые 9 изображений
        # Display the first 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # uint8 - Преобразуем изображение в числовой вид 0-255.
        # uint8 - Convert image to uint8 (0-255)
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Автоматически выбирает оптимальный размер буфера для предвыборки данных, в зависимости от доступных ресурсов и устройства
# Automatically selects the optimal buffer size for data prefetching based on available resources and device
AUTOTUNE = tf.data.AUTOTUNE
# Кэширование и предвыборка данных для ускорения обучения
# Caching and prefetching data for faster training
train_ds = train_ds.cache().prefetch(
    buffer_size=AUTOTUNE)  # данные из обучающего набора сохраняются в кэш после первой загрузки.
    # Data from the training set is cached after the first load.
# подготовка данных для следующей итерации, пока модель работает с текущими данными
# Preparing data for the next iteration while the model works with current data
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Количество классов
# Number of classes
num_classes = 2

# Строим модель, определяя слои
# Building the model by defining layers
model = keras.Sequential([
    # Нормализация изображений (0-1)
    # Normalizing images (0-1)
    keras.layers.Rescaling(1. / 255),
    # Свёрточный слой
    # Convolutional layer
    keras.layers.Conv2D(32, 3, activation='relu'),
    #  Выбираем максимальные значения из карты признаков(полученную из свёрточного слоя)
    # Selecting the maximum values from the feature map (obtained from the convolutional layer)
    keras.layers.MaxPooling2D(pool_size=(4, 4)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3, 3)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes),
])

# Оптимизатор для изменения весов модели и корректировки скорости обучения для каждого параметра.
# Optimizer to update model weights and adjust learning rate for each parameter.
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Настраиваем модель
# Configuring the model
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# # Определяем callback EarlyStopping
# # Defining EarlyStopping callback
# early_stopping = EarlyStopping(
#     monitor='val_loss',  # Отслеживаем ошибку на валидации
#     # Tracking validation loss
#     patience=5,  # Остановить через 5 эпох без улучшений
#     # Stop after 5 epochs without improvements
#     restore_best_weights=True  # Восстановить веса модели с лучшей метрикой
#     # Restore model weights with the best metric
# )

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=19,
    # callbacks=[early_stopping] # Добавляем EarlyStopping в качестве callback'а для обучения модели.
    # Adding EarlyStopping as a callback
)
test_loss, test_acc = model.evaluate(val_ds)
print(f'Точность модели: {test_acc * 100:.2f}')
# Model accuracy

# Построение графиков
# Plotting graphs
plt.figure(figsize=(12, 5))

# Потери
# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Эпохи')  # Epochs
plt.ylabel('Потери')  # Loss
plt.title('График потерь')  # Loss graph
plt.legend()

# Точность
# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Эпохи')  # Epochs
plt.ylabel('Точность')  # Accuracy
plt.title('График точности')  # Accuracy graph
plt.legend()
plt.show()

y_true = []
y_pred = []

# Сбор истинных и предсказанных меток
# Collecting true and predicted labels
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Построение матрицы ошибок
# Plotting the confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='viridis', xticks_rotation=45)
plt.title("Матрица ошибок")  # Confusion matrix
plt.show()

# Примеры предсказаний
# Examples of predictions
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):  # Берем один пакет изображений
    # Take one batch of images
    predictions = model.predict(images)
    for i in range(9):  # Отобразим первые 9 изображений
        # Display the first 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[labels[i]]
        plt.title(f"Предсказание: {predicted_class}\nИстина: {true_class}")
        # Prediction: {predicted_class}\nTrue: {true_class}
        plt.axis("off")
plt.show()
