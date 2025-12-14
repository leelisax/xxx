# RNN для генерации текста на основе обучающего набора.

## ЗАДАНИЕ ИЗ МЕТОДИЧКИ

### Задание 7: RNN для генерации текста

Задача: создать рекуррентную нейронную сеть (RNN) для генерации текста на основе обучающего набора. 
 
Требования: 
Использовать LSTM ячейки (128 units) 
Embedding layer для векторизации символов 
Предсказание следующего символа 
Управление температурой при генерации 
 
Код-заготовка (Python): 
```
import tensorflow as tf 
 
class TextGenerator: 
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=128): 
        # TODO: Создать модель с Embedding + LSTM + Dense слоями 
        # Архитектура: 
        # - Embedding(vocab_size, embedding_dim) 
        # - LSTM(lstm_units, return_sequences=True) 
44 
 
        # - LSTM(lstm_units) 
        # - Dense(vocab_size, activation='softmax') 
        self.model = None 
        self.vocab_size = vocab_size 
     
    def preprocess_text(self, text, sequence_length=40): 
        # TODO: Преобразовать текст в последовательности 
        # Создать mapping символ -> индекс 
        # Генерировать training pairs (input_sequence, target_char) 
        pass 
     
    def compile_and_train(self, X_train, y_train, epochs=50): 
        # TODO: Обучить модель 
        # Loss: sparse_categorical_crossentropy 
        # Optimizer: Adam 
        pass 
     
    def generate_text(self, seed_text, num_chars=100, temperature=1.0): 
        # TODO: Генерировать текст из seed 
        # Параметр temperature контролирует случайность 
        # temperature < 1: более предсказуемо 
        # temperature > 1: более разнообразно 
        # 1. Закодировать seed_text 
        # 2. Цикл: 
        #    - Предсказать вероятности следующего символа 
        #    - Применить temperature: P = P^(1/T) 
        #    - Выбрать символ по распределению 
        #    - Добавить в последовательность 
        pass 
``` 
## Что нужно дополнить: 
### 1. Слои Embedding и LSTM 
### 2. Функцию preprocess_text с character-level encoding 
### 3. Обучение модели с callback для мониторинга 
### 4. Реализацию temperature-based sampling 
### 5. Экспериментирование с разными значениями temperature 

# 2. АЛГОРИТМ РАБОТЫ НС ПО БЛОКАМ

## Блок 0: Импорт библиотек

```
python
import numpy as np, tensorflow as tf, keras, matplotlib.pyplot as plt
```

### Что делает: Подключает TensorFlow/Keras для нейросети, NumPy для массивов, Matplotlib для графиков. Проверяет версию TF.

## Блок 1: Подготовка данных IMDB

```
python
vocab_size = 10000  # топ-10k слов
max_len = 100       # макс. длина отзыва
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
```

### Что делает:
**- Загружает готовый датасет IMDB (25k отзывов на train/test)**
**- Урезает до 15k/5k примеров + max_len=100 → быстрее обучение**
***```pad_sequences``` дополняет короткие отзывы нулями до длины 100***

***Результат: ```x_train.shape = (15000, 100)``` — 15k отзывов по 100 слов.***

## Блок 2: Positional Encoding

```
python
class PositionalEncoding(layers.Layer):
    # pe = sin(position / 10000^(2i/d)) для чётных индексов
    # pe = cos(position / 10000^(2i/d)) для нечётных
```

### Что делает: Добавляет информацию о позиции слов в последовательность (Transformer не видит порядок слов без этого).

***pe — фиксированная матрица синусов/косинусов размером (1, max_len, embedding_dim)***

***Складывается с embedding'ами: x + pe[:seq_len]***

## Блок 3: Multi-Head Self-Attention

```
python
class MultiHeadSelfAttention(layers.Layer):
    Q = Wq(x), K = Wk(x), V = Wv(x)  # 3 проекции
    attention = softmax(QK^T / √d_k) * V  # scaled dot-product
```

### Что делает: Сердце Transformer'а — вычисляет, какие слова в предложении важны друг для друга.

**- Разбивает embedding на num_heads=4 "голов"**

**- Для каждой головы: Q·K^T / √d_k → веса внимания → умножение на V**

**- Конкатенирует головы и проецирует через Dense.**

***Маска отключает padding-нули.***

## Блок 4: FeedForward + EncoderLayer

```
python
class FeedForward: Dense(hidden_dim) → ReLU → Dense(embedding_dim)
class EncoderLayer: 
    x1 = LayerNorm(x + Dropout(MultiHeadAttention(x)))
    x2 = LayerNorm(x1 + Dropout(FFN(x1)))
```

### Что делает:

**- FFN — простая MLP для нелинейности после attention**

**- EncoderLayer — один блок Transformer'а:**

***Self-attention + residual connection + LayerNorm***

***FeedForward + residual connection + LayerNorm***

**- Dropout предотвращает переобучение**

## Блок 5: Сборка модели

```
python
inputs → Embedding(10000→64) → PosEnc → 
EncoderLayer(1 слой) → GlobalAvgPool → 
Dense(64) → Dense(1, sigmoid)
```

### Архитектура (параметры из задания):

```
text
embedding_dim=64, num_heads=4, hidden_dim=128, num_layers=1
```

**- Embedding: слова → векторы 64D**

**- PosEnc: добавляет позицию**

**- 1 EncoderLayer: attention + FFN**

**- GlobalAvgPool: усредняет по последовательности → 1 вектор**

**- Классификатор: 64 → 1 (sigmoid для бинарной классификации)**

## Блок 6: Обучение

```
python
model.compile(loss="binary_crossentropy", Adam(lr=1e-3), metrics=["accuracy"])
history = model.fit(..., epochs=5, batch_size=128)
```

### Что делает:

**- Loss: Binary CrossEntropy (0=negative, 1=positive отзыв)**

**- Оптимизатор: Adam с lr=0.001**

**- 5 эпох на урезанных данных = ~2 минуты в Colab**

**- validation_split=0.2 — 20% train на валидацию**

## Блок 7-8: Оценка + Визуализация

```
python
test_acc = model.evaluate(x_test, y_test)  # ~0.70
plt.plot(history["accuracy"], history["val_accuracy"])  # графики
```

### Выводит:

**- Test accuracy (~70%)**

**- Графики: train/val accuracy/loss по эпохам**

## Блок 9: Демонстрация работы

```
python
def decode_review(encoded): " ".join(index_word[i] for i in encoded)
model.predict(sample) → "positive" если >0.5
```

### Что делает:

**- Декодирует числа → слова (словарь IMDB)**

**- Показывает 3 примера: текст отзыва + предсказание модели**


# 3. ОТВЕТ НА КОНТРОЛЬНЫЙ ВОПРОС


## 7. Как выбрать оптимальный размер блока для алгоритмов во внешней памяти? 


## Оптимальный размер блока для внешней памяти

Размер блока **блокирует алгоритмы** (внешняя сортировка, хэш-таблицы, B-деревья) выбирается по формуле, максимизирующей **I/O-параллелизм** и минимизирующей количество обращений к диску.[1]

## Основная формула
```
B_opt = √(M² / N)   или   B_opt ≈ M / √(N/M)
Где:
M — размер памяти (в байтах)
N — размер данных (в байтах) 
B — размер блока (в байтах)
```

**Принцип:** Блок должен помещаться **2-3 раза** в память для эффективного **double buffering**.


## Таблица оптимальных размеров

| Алгоритм | M=1GB | M=4GB | M=16GB | Рекомендация |
|----------|-------|-------|--------|--------------|
| MergeSort | 1-4KB | 4-8KB | 8-16KB | 4KB, 8KB, 16KB |
| Hash Join | 64-128MB | 128-256MB | 256-512MB | 2^n MB |
| Index Scan | 64KB | 128KB | 256KB | Страница диска |
| B+-дерево | 4KB | 8KB | 16KB | Фикс. страница |


## Вывод
**Золотое правило:** `B ≈ √(M²/N)` + округление до **стандартного размера** (4/8/16/64KB, 128/256MB). Это даёт **оптимальный баланс** между I/O-параллелизмом и использованием памяти.[1]
