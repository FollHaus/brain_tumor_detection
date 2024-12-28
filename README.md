### README для проекта классификации болезни мозга с использованием TensorFlow и Keras

# Классификация изображений с использованием сверточных нейронных сетей (CNN) на базе TensorFlow

Этот проект включает в себя создание и обучение модели для классификации изображений с использованием сверточных нейронных сетей (CNN) на библиотеке TensorFlow и Keras. Данные для обучения загружаются с использованием метода `image_dataset_from_directory`, а для улучшения процесса обучения используются техники предобработки, такие как нормализация и кэширование.

Данные были взяты с Kaggle: https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri?resource=download&select=Dataset
---

## Описание

Проект реализует классификацию изображений, используя сверточную нейронную сеть (CNN), и включает в себя:
1. **Загрузку и подготовку данных.**
2. **Построение модели.**
3. **Обучение модели.**
4. **Оценку производительности модели с использованием различных метрик.**
5. **Визуализацию результатов, таких как потери и точность, а также матрицы ошибок.**

Модель настроена для работы с двумя классами, но её можно адаптировать для большего количества классов.

---

## Требования

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn

Для установки зависимостей используйте `pip`:
```
pip install tensorflow matplotlib scikit-learn
```

---

## Структура проекта

```
/dataset               # Папка с изображениями для обучения и валидации
/brain_tumor_detect.py # Основной файл с кодом
README.md              # Этот файл
```

---

## Описание кода

### 1. Загрузка данных с диска 

```python
train_ds = keras.utils.image_dataset_from_directory(
    data_set,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
```

**Параметры:**
- `data_set`: Путь к каталогу с изображениями для обучения.
- `validation_split=0.2`: Делит данные на обучающую и валидационную выборки. 80% для обучения, 20% для валидации.
- `subset="training"`: Указывает, что данные будут использоваться для обучения.
- `seed=123`: Сеед для воспроизводимости случайных разбиений.
- `image_size=(img_height, img_width)`: Размер изображений, к которому они будут приведены перед загрузкой.
- `batch_size=batch_size`: Размер пакета данных.

Аналогично, данные для валидации загружаются с параметром `subset="validation"`.

---

### 2. Подсчет распределения классов и просмотр изображений

```python
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
```
- `Counter(labels)` подсчитывает количество изображений каждого класса в обучающем наборе данных. Эти данные используются для создания столбчатой диаграммы, показывающей распределение классов.

На графике видно, что имеется разбалансировка данных.

![image](https://github.com/user-attachments/assets/211833b0-220a-4833-af09-dfefbef29a3c)


#### Посмотрим что представляют из себя изображения

```python
# Просмотр первых 9 изображений из train_ds
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):  # Берем один пакет изображений
    for i in range(9):  # Отобразим первые 9 изображений
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # uint8 - Преобразуем изображение в числовой вид 0-255.
        plt.title(class_names[labels[i]])
        plt.axis("off")
```

#### Tumor - опухоль. Healthy - здоров
 
![image](https://github.com/user-attachments/assets/a0633f38-f546-4e61-86fe-411dcb2cf5b3)

---

### 3. Построение модели

#### Настройка для кэширования и асинхронной подгрузки данных

```python
# Автоматически выбирает оптимальный размер буфера для предвыборки данных, в зависимости от доступных ресурсов и устройства
AUTOTUNE = tf.data.AUTOTUNE
# Кэширование и предвыборка данных для ускорения обучения
train_ds = train_ds.cache().prefetch(
    buffer_size=AUTOTUNE)  # данные из обучающего набора сохраняются в кэш после первой загрузки.
# подготовка данных для следующей итерации, пока модель работает с текущими данными
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```
В чем разница между `.cache()` и `.prefetch()`?

- **cache()** сохраняет обработанные данные, чтобы не читать их повторно из источника (например, с диска).
- **prefetch()** позволяет загружать данные параллельно с обучением модели, снижая время простоя процессора или GPU.


```python
model = keras.Sequential([
    keras.layers.Rescaling(1. / 255),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(4, 4)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3, 3)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes),
])
```

**Слои:**
1. **Rescaling(1./255)**: Нормализует значения пикселей изображений, деля их на 255, чтобы они находились в диапазоне от 0 до 1.
2. **Conv2D(32, 3, activation='relu')**: Свёрточный слой, который применяет 32 фильтра размером 3x3 для извлечения карты признаков(края, текстуры, форма). Функция активации `ReLU` помогает нейронной сети обучаться более эффективно.
3. **MaxPooling2D(pool_size=(4, 4))**: Слой максимального объединения, который уменьшает размерность выходных данных, выбирая максимальное значение из области 4x4.
4. **Flatten()**: Преобразует многомерные данные после свёртки в одномерный вектор.
5. **Dense(128, activation='relu')**: Полносвязный слой с 128 нейронами и функцией активации `ReLU`.
6. **Dense(num_classes)**: Выходной слой с количеством нейронов, равным числу классов.
   
**MaxPooling2D()** - работает так, допустим свёрточный слой вернул нам карту признаков 4х4 и мы указали пулинг с параметром 2х2.

Карта признаков свёрточного слоя:
```
1   3   2   4
5   6   7   8
9   8   6   4
1   2   3   5  
```
Пулинг берет максимальное значение из каждого блока к примеру 1-й блок = [1 3 2 4] значение будет равно 4, тем самым пулинг вернет нам новую карту признаков 2х2.

Новая карта признаков:
```
4 8
9 5
```
---

### 4. Создание и настройка модели

```python
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
```

**Параметры:**
- `optimizer=optimizer`: Оптимизатор, используемый для обучения модели. В данном случае используется `Adam`.
- `loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)`: Функция потерь для многоклассовой классификации с использованием меток в виде целых чисел (не one-hot encoded).
- `metrics=['accuracy']`: Метрика для оценки точности модели.

---

### 5. Обучение модели

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=19)
```

**Параметры:**
- `train_ds`: Тренировочный набор данных.
- `validation_data=val_ds`: Валидационный набор данных.
- `epochs=19`: Количество эпох для обучения модели.

---

### 6. Оценка модели

```python
test_loss, test_acc = model.evaluate(val_ds)
print(f'Точность модели: {test_acc * 100:.2f}')
```
![image](https://github.com/user-attachments/assets/f5b99576-c37a-4a07-a35b-3e3ed23f3044)

**Метод:**
- `evaluate(val_ds)` вычисляет ошибку и точность на валидационном наборе данных.

---

### 7. Построение графиков потерь и точности обучения и валидации.

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.title('График потерь')
plt.legend()
```

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.title('График точности')
plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/cf533b55-1e96-4860-973b-7d048dc7af08)

---

### 8. Построение матрицы ошибок

```python
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='viridis', xticks_rotation=45)
```

**Параметры:**
- `confusion_matrix(y_true, y_pred)` создает матрицу ошибок, которая сравнивает истинные метки и предсказанные.
- `ConfusionMatrixDisplay` используется для визуализации матрицы ошибок.

Посмотрим где ошибается модель:

![image](https://github.com/user-attachments/assets/7bde716e-b875-4ecc-a8b6-b7964458eb82)

---

### 9. Примеры предсказаний

```python
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
```

- Отображаем изображения с их истинными и предсказанными метками.
- 
![image](https://github.com/user-attachments/assets/a0d7f1a9-3322-40bb-9f44-f866b03454e7)

---

## Заключение

Этот проект является примером использования сверточных нейронных сетей для классификации опухоли мозга. Модель настраивается и обучается с использованием стандартных методов в TensorFlow/Keras, таких как нормализация, свёртки и объединение. Для улучшения точности можно применять различные методы, включая изменение архитектуры модели, добавление регуляризации или использование методов аугментации данных.

Также в коде есть закомментированный код:

**Код предотвращает переобучение модели и останавливает её обучение, когда она перестаёт улучшаться.**

```python
# Определяем callback EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Отслеживаем ошибку на валидации
    patience=5,  # Остановить через 5 эпох без улучшений
    restore_best_weights=True  # Восстановить веса модели с лучшей метрикой
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=19,
    # callbacks=[early_stopping] # Добавляем EarlyStopping в качестве callback'а для обучения модели.
)
```

