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
data_set = Path('./dataset')

# Тренировочная выборка
train_ds = keras.utils.image_dataset_from_directory(
    data_set,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# Валидационная выборка
val_ds = keras.utils.image_dataset_from_directory(
    data_set,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
class_names = train_ds.class_names

# Метки
labels = []
for _, label_batch in train_ds:  # Пакет данных и меток
    labels.extend(label_batch.numpy())  # Добавление меток в список

# Подсчет количества каждого класса с помощью Counter
class_counts = Counter(labels)

# Создание столбчатого графика и смотрим распределение классов в train_ds
plt.figure(figsize=(10, 9))
plt.bar(class_names, [class_counts[i] for i in range(len(class_names))], color='skyblue')
plt.xlabel('Классы', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.title('Распределение классов в train_ds', fontsize=16)
plt.xticks(rotation=45)
plt.show()

# Просмотр первых 9 изображений из train_ds
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):  # Берем один пакет изображений
    for i in range(9):  # Отобразим первые 9 изображений
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # uint8 - Преобразуем изображение в числовой вид 0-255.
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Автоматически выбирает оптимальный размер буфера для предвыборки данных, в зависимости от доступных ресурсов и устройства
AUTOTUNE = tf.data.AUTOTUNE
# Кэширование и предвыборка данных для ускорения обучения
train_ds = train_ds.cache().prefetch(
    buffer_size=AUTOTUNE)  # данные из обучающего набора сохраняются в кэш после первой загрузки.
# подготовка данных для следующей итерации, пока модель работает с текущими данными
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Количество классов
num_classes = 2

# Строим модель, определяя слои
model = keras.Sequential([
    # Нормализация изображений (0-1)
    keras.layers.Rescaling(1. / 255),
    # Свёрточный слой
    keras.layers.Conv2D(32, 3, activation='relu'),
    #  Выбираем максимальные значения из карты признаков(полученную из свёрточного слоя)
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
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Настраиваем модель
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# # Определяем callback EarlyStopping
# early_stopping = EarlyStopping(
#     monitor='val_loss',  # Отслеживаем ошибку на валидации
#     patience=5,  # Остановить через 5 эпох без улучшений
#     restore_best_weights=True  # Восстановить веса модели с лучшей метрикой
# )

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=19,
    # callbacks=[early_stopping] # Добавляем EarlyStopping в качестве callback'а для обучения модели.
)
test_loss, test_acc = model.evaluate(val_ds)
print(f'Точность модели: {test_acc * 100:.2f}')

# Построение графиков
plt.figure(figsize=(12, 5))

# Потери
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.title('График потерь')
plt.legend()

# Точность
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.title('График точности')
plt.legend()
plt.show()

y_true = []
y_pred = []

# Сбор истинных и предсказанных меток
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Построение матрицы ошибок
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='viridis', xticks_rotation=45)
plt.title("Матрица ошибок")
plt.show()

# Примеры предсказаний
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):  # Берем один пакет изображений
    predictions = model.predict(images)
    for i in range(9):  # Отобразим первые 9 изображений
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[labels[i]]
        plt.title(f"Предсказание: {predicted_class}\nИстина: {true_class}")
        plt.axis("off")
plt.show()
