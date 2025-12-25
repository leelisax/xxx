<img width="3243" height="801" alt="image" src="https://github.com/user-attachments/assets/ce35a6d8-47a0-42b5-b347-742aee3360b8" />Нейронная сеть для определения качества пищи по химическим показателям Галкина, Железная

## Актуальность 
В современном мире контроль качества продуктов питания становится критически важным из-за усложнения цепочек поставок и роста использования синтетических добавок. 
- Минимизация человеческого фактора при анализе сложных химических составов.
- Мгновенная классификация продуктов на пригодные и непригодные.
- Снижение затрат на лабораторные исследования за счет предварительной цифровой оценки.

## Датасет
Общие сведения:
Объект наблюдения: отдельный образец вина (партия/бутылка).​
Каждая строка описывает одно вино, столбцы содержат результаты лабораторных измерений и экспертную оценку качества.​
Физико‑химические признаки:
fixed acidity — фиксированная кислотность, в основном винные кислоты (числовой, непрерывный признак).​
volatile acidity — летучая кислотность, преимущественно уксусная кислота (числовой).​
citric acid — содержание лимонной кислоты, влияющей на свежесть и вкус вина (числовой).​
residual sugar — остаточный сахар, количество сахара после брожения (числовой).​
chlorides — концентрация хлоридов (в основном NaCl) в вине (числовой).​
free sulfur dioxide — свободный диоксид серы, не связанный с другими молекулами (числовой).​
total sulfur dioxide — общий диоксид серы (сумма свободной и связанной форм) (числовой).​
density — плотность вина, связана с содержанием сахара и алкоголя (числовой).​
pH — показатель кислотности вина (числовой).​
sulphates — концентрация сульфатов, влияющих на консервацию и вкус (числовой).​
alcohol — объёмная доля алкоголя в вине (числовой).​
Целевой признак качества:
quality — органолептическая оценка качества вина экспертами по шкале от 0 (очень плохо) до 10 (отлично); в проекте дополнительно агрегируется в три класса: низкое, среднее и высокое качество.​




## Обзор программы

### Назначение
Программа решает задачу **бинарной классификации** для определения качества вина:
- **Входные данные**: 11 физико-химических параметров (алкоголь, кислотность, сахар и т.д.)
- **Выходные данные**: Класс качества (Низкое ≤6 / Высокое >6)
- **Датасет**: UCI Wine Quality (комбинированный красное + белое вино = 6,497 образцов)

### Архитектура решения
```
Загрузка данных
      ↓
Анализ и визуализация
      ↓
Предобработка (нормализация)
      ↓
Разделение на train/test
      ↓
┌─────────────────────────────────────┐
│  Обучение 5 алгоритмов параллельно  │
├─────────────────────────────────────┤
│ 1. TensorFlow Neural Network (основная)
│ 2. Logistic Regression
│ 3. Random Forest
│ 4. XGBoost
│ 5. LightGBM
└─────────────────────────────────────┘
      ↓
Сравнение результатов
      ↓
Визуализация (графики, ROC-кривые)
```

---

## 🔧 Этап 1: Установка и импорт

### Строка 1-2: Установка библиотек
```python
!pip install -q tensorflow scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm
```

**Назначение**: Установить необходимые Python-библиотеки
- `-q` флаг = "quiet" (выводить меньше информации)

**Библиотеки**:
- **tensorflow**: Фреймворк глубокого обучения от Google (для нейросетей)
- **scikit-learn**: Классический ML (логрегрессия, случайный лес, метрики)
- **pandas**: Работа с табличными данными (DataFrame)
- **numpy**: Математические операции с массивами
- **matplotlib/seaborn**: Визуализация графиков
- **xgboost/lightgbm**: Градиентный бустинг

### Строки 3-13: Импорт модулей
```python
import numpy as np                              # Численные операции
import pandas as pd                             # Работа с данными
import matplotlib.pyplot as plt                 # Графики
import seaborn as sns                           # Красивые графики
from sklearn.model_selection import train_test_split  # Разделение данных
from sklearn.preprocessing import StandardScaler      # Нормализация
from sklearn.metrics import (                        # Метрики оценки
    accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier    # Случайный лес
from sklearn.linear_model import LogisticRegression   # Логистическая регрессия
from xgboost import XGBClassifier                      # XGBoost
from lightgbm import LGBMClassifier                    # LightGBM
import tensorflow as tf                                # TensorFlow
from tensorflow import keras                           # Keras API
from tensorflow.keras import layers, regularizers      # Слои и регуляризация
import warnings
warnings.filterwarnings('ignore')                      # Скрыть предупреждения
```

**Что импортируем**:

| Модуль | Функция | Пример использования |
|--------|---------|----------------------|
| `train_test_split` | Разделить данные 80/20 | `X_train, X_test = train_test_split(X, test_size=0.2)` |
| `StandardScaler` | Нормализация (μ=0, σ=1) | `scaler.fit_transform(X)` |
| `accuracy_score` | Точность предсказаний | `accuracy_score(y_true, y_pred)` |
| `RandomForestClassifier` | 200 деревьев решений | `rf.fit(X_train, y_train)` |
| `XGBClassifier` | Экстремальный градбустинг | `xgb.fit(X_train, y_train)` |
| `keras.Sequential` | Последовательная сеть | Слой → слой → выход |

### Строки 14-17: Настройка визуализации
```python
plt.rcParams['figure.figsize'] = (14, 8)    # Размер фигур по умолчанию
sns.set_style("whitegrid")                   # Стиль графиков (белый с сеткой)
plt.rcParams['font.size'] = 11               # Размер шрифта 11pt
```

**Эффект**: Все последующие графики будут большими (14x8), красивыми и разборчивыми.

---

## 📥 Этап 2: Загрузка датасета

### Строки 1-2: Определение URL датасета
```python
url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
```

**Источник**: UCI Machine Learning Repository (официальный репозиторий ML датасетов)
- **Красное вино**: 1,599 образцов
- **Белое вино**: 4,898 образцов

### Строки 3-7: Загрузка файлов
```python
df_red = pd.read_csv(url_red, sep=';')      # Загрузить красное вино
df_red['wine_type'] = 'red'                 # Добавить колонку с типом

df_white = pd.read_csv(url_white, sep=';')  # Загрузить белое вино
df_white['wine_type'] = 'white'             # Добавить колонку с типом
```

**Детали**:
- `sep=';'` - используется точка с запятой как разделитель (европейский формат CSV)
- Добавляем новую колонку `wine_type` для отслеживания типа вина

**Структура DataFrame после загрузки**:
```
   fixed acidity  volatile acidity  citric acid  ... alcohol  quality  wine_type
0           7.4                0.70         0.00  ...   9.4       5      red
1           7.8                0.88         0.00  ...   9.8       5      red
...
1599        5.9                0.62         0.20  ...  10.2       6      red
1600        8.8                0.47         0.04  ...  10.6       6      white
...
```

### Строка 8: Объединение датасетов
```python
df = pd.concat([df_red, df_white], ignore_index=True)
```

**Операция**: Вертикальное объединение
- Красное вино (строки 0-1599) + Белое вино (строки 1600-6497) = 6,497 образцов
- `ignore_index=True` перенумеровывает индексы (0, 1, 2, ..., 6496)

### Строки 9-12: Вывод информации
```python
print(f"✓ Датасет загружен!")
print(f"  Форма: {df.shape}")                    # (6497, 13)
print(f"  Красного вина: {len(df_red)}")        # 1599
print(f"  Белого вина: {len(df_white)}")        # 4898
```

**df.shape** возвращает кортеж (строки, столбцы) = (6497, 13)

---

## 📊 Этап 3: Анализ данных

### Строки 1-2: Распределение качества
```python
print("Распределение качества:")
print(df['quality'].value_counts().sort_index())
```

**Выходной результат**:
```
quality
3      10
4     389
5    1457
6    2198
7    1840
8     193
9      18
```

**Интерпретация**: Большинство вин имеют оценку 5-7, класс 3 и 9 редкие.

### Строки 3-15: Построение гистограмм
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 строка, 2 столбца

# Левый график: Общее распределение
axes[0].hist(df['quality'], bins=20, color='#667eea', alpha=0.7, edgecolor='black')
axes[0].set_title('Распределение качества вина', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Качество')
axes[0].set_ylabel('Количество')

# Правый график: По типам вина
df_red['quality'].hist(bins=15, alpha=0.6, label='Red', ax=axes[1], color='#e74c3c')
df_white['quality'].hist(bins=15, alpha=0.6, label='White', ax=axes[1], color='#f1c40f')
axes[1].set_title('Распределение по типам вина', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Параметры графиков**:
- `bins=20` - количество столбцов в гистограмме
- `color='#667eea'` - hex-код цвета (фиолетовый)
- `alpha=0.7` - прозрачность (70% непрозрачность)
- `edgecolor='black'` - черная граница столбцов
- `tight_layout()` - автоматически упаковать субплоты

**Что видим**: 
- Нормальное распределение, пик в районе 5-6
- Красное вино: более низкое качество в среднем
- Белое вино: более высокое качество в среднем

---

## 🔄 Этап 4: Предобработка

### Строка 1: Удаление дубликатов
```python
df = df.drop_duplicates().reset_index(drop=True)
```

**Операции**:
- `drop_duplicates()` удаляет полностью идентичные строки
- `reset_index(drop=True)` переиндексирует (0, 1, 2, ...) и удаляет старый индекс

**Пример**: Если было 2 идентичных образца вина, остается 1.

### Строки 2-3: Создание целевой переменной (бинарная классификация)
```python
df['quality_binary'] = (df['quality'] > 6).astype(int)
```

**Преобразование**:
```
quality: 3, 4, 5, 6, 7, 8, 9     (исходные оценки 3-9)
                    ↓ > 6?
quality_binary: 0, 0, 0, 0, 1, 1, 1  (0=низкое, 1=высокое)
```

**Распределение классов** (пример):
- Класс 0 (≤6): 3,937 образцов (60.6%)
- Класс 1 (>6): 2,560 образцов (39.4%)

### Строки 4-8: Выделение признаков
```python
X = df.drop(columns=['quality', 'quality_binary', 'wine_type'])
y = df['quality_binary']
```

**Результат**:
- `X` (11 признаков): fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- `y` (целевая): 0 или 1

**Форма**: `X.shape = (6497, 11)`, `y.shape = (6497,)`

### Строки 9-12: Вывод списка признаков
```python
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")
```

**Вывод**:
```
1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
...
11. alcohol
```

### Строки 13-15: Разделение на обучающую и тестовую выборки
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Параметры**:
- `test_size=0.2` - 20% на тестирование, 80% на обучение
- `random_state=42` - зафиксировать случайность (для воспроизводимости)
- `stratify=y` - сохранить пропорцию классов в обеих выборках

**Результат**:
- `X_train.shape = (5197, 11)` - обучение
- `X_test.shape = (1300, 11)` - тестирование
- Класс 0: 60.6% в обеих выборках
- Класс 1: 39.4% в обеих выборках

### Строки 16-17: Нормализация данных
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Что это делает**:

```
StandardScaler преобразует: x_new = (x - mean) / std_dev

Пример (для признака 'alcohol'):
Исходные: [9.4, 9.8, 10.2, ...]  (среднее ≈ 10.5, стд ≈ 1.2)
             ↓
Масштабированные: [-0.92, 0.25, 1.42, ...]  (среднее ≈ 0, стд ≈ 1)
```

**Почему нужна нормализация**:
1. **Равный вклад признаков**: Алкоголь (0-14) и pH (2.7-4.0) имеют разные масштабы
2. **Скорость сходимости**: Нейросеть обучается быстрее
3. **Численная стабильность**: Предотвращает переполнение/переливание

**Критично**: 
- Используем `.fit()` на тренировочных данных
- Затем `.transform()` на тестовых (так как мы не знаем статистику тестовых данных)

---

## 🧠 Этап 5: Построение нейросети

### Строки 1-8: Вывод параметров
```python
print("\n📋 Параметры нейросети:")
print("  input_dim: 11")
print("  hidden_layers: [128, 64, 32]")
print("  activation: relu")
print("  dropout_rate: 0.3")
print("  l2_reg: 1e-4")
print("  learning_rate: 0.001")
print("  batch_size: 32")
print("  epochs: 150")
```

**Параметры**:
- **input_dim**: 11 нейронов входа (по числу признаков)
- **hidden_layers**: 3 скрытых слоя с 128, 64, 32 нейронами
- **activation**: ReLU активация (max(0, x))
- **dropout_rate**: 30% нейронов случайно "выключаются"
- **l2_reg**: L2 регуляризация с λ=1e-4 (штраф за большие веса)
- **learning_rate**: 0.001 = как быстро обновляются веса
- **batch_size**: 32 образца за раз
- **epochs**: максимум 150 итераций (но ранняя остановка может остановить раньше)

### Строки 9-18: Определение архитектуры модели
```python
model = keras.Sequential([
    layers.Input(shape=(11,)),
    layers.Dense(128, activation='relu', 
                kernel_regularizer=regularizers.l2(1e-4), name='hidden_1'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', 
                kernel_regularizer=regularizers.l2(1e-4), name='hidden_2'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', 
                kernel_regularizer=regularizers.l2(1e-4), name='hidden_3'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid', name='output')
])
```

**Архитектура (слой за слоем)**:

```
┌─────────────────────────────────────┐
│ Input Layer: 11 нейронов            │
│ (11 физико-химических параметров)  │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Dense Layer 1: 128 нейронов         │
│ ├─ Функция активации: ReLU          │
│ ├─ Regularizer: L2 (1e-4)           │
│ ├─ Параметры: 11×128 + 128 = 1,536  │
│ └─ Название: 'hidden_1'             │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Dropout Layer: 30% отключения       │
│ (случайно выключаем 38 из 128)      │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Dense Layer 2: 64 нейрона           │
│ ├─ Функция активации: ReLU          │
│ ├─ Regularizer: L2 (1e-4)           │
│ ├─ Параметры: 128×64 + 64 = 8,256   │
│ └─ Название: 'hidden_2'             │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Dropout Layer: 30% отключения       │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Dense Layer 3: 32 нейрона           │
│ ├─ Функция активации: ReLU          │
│ ├─ Regularizer: L2 (1e-4)           │
│ ├─ Параметры: 64×32 + 32 = 2,080    │
│ └─ Название: 'hidden_3'             │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Dropout Layer: 30% отключения       │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Output Layer: 1 нейрон              │
│ ├─ Функция активации: Sigmoid       │
│ ├─ Параметры: 32×1 + 1 = 33         │
│ ├─ Выход: значение между 0 и 1      │
│ ├─ Интерпретация:                   │
│ │  < 0.5 → класс 0 (низкое)        │
│ │  ≥ 0.5 → класс 1 (высокое)       │
│ └─ Название: 'output'               │
└─────────────────────────────────────┘

ВСЕГО ПАРАМЕТРОВ: 1,536 + 8,256 + 2,080 + 33 ≈ 11,900
```

**Подробнее о компонентах**:

1. **Dense (полносвязный слой)**:
   ```
   y = activation(W × x + b)
   
   где W - матрица весов, b - смещение
   
   Пример Dense(128):
   - Входит: вектор из 11 чисел
   - Выходит: вектор из 128 чисел
   - Параметров: 11×128 (веса) + 128 (смещения)
   ```

2. **ReLU активация**:
   ```
   ReLU(x) = max(0, x)
   
   -5  →  0
   -1  →  0
    0  →  0
    1  →  1
    5  →  5
   
   Эффект: Вносит нелинейность, позволяет сети моделировать сложные функции
   ```

3. **Dropout 0.3**:
   ```
   На каждой эпохе обучения:
   - Генерируем случайное число [0,1) для каждого нейрона
   - Если < 0.3 → выключаем нейрон (выход = 0)
   - Если ≥ 0.3 → оставляем включенным
   
   Эффект: Предотвращает переобучение ("взаимная адаптация")
   Во время тестирования: все нейроны включены
   ```

4. **Sigmoid активация (выход)**:
   ```
   Sigmoid(x) = 1 / (1 + e^(-x))
   
   Свойства:
   - Выходит значение между 0 и 1
   - Идеален для вероятности класса
   - Гладкая функция (хорошие градиенты)
   ```

5. **L2 Регуляризация**:
   ```
   Loss_total = Loss_original + λ × Σ(w²)
   
   где λ = 1e-4 = 0.0001
   
   Эффект: Штрафует большие веса
   - Вынуждает веса быть маленькими
   - Предотвращает переобучение
   - Улучшает обобщение
   ```

### Строки 19-23: Компилирование модели
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Компилирование** = подготовка к обучению:

- **optimizer='Adam'**:
  ```
  Адаптивный момент оценки (Adam = Adaptive Moment Estimation)
  
  Как работает:
  1. Вычислить градиент: ∇L = ∂L/∂w
  2. Обновить вес: w_new = w_old - lr × ∇L
  
  Особенность Adam:
  - Использует экспоненциально взвешенное среднее градиентов
  - Адаптивная скорость обучения для каждого параметра
  - Быстрая сходимость в большинстве случаев
  
  learning_rate=0.001:
  - Если градиент = 0.1 → обновление = 0.1 × 0.001 = 0.0001
  - Баланс между скоростью и стабильностью
  ```

- **loss='binary_crossentropy'**:
  ```
  Функция ошибки для бинарной классификации:
  
  loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
  
  где:
  - y = истинный класс (0 или 1)
  - ŷ = предсказанная вероятность
  
  Интерпретация:
  - Если y=1, ŷ=0.9: loss ≈ -log(0.9) ≈ 0.11 (хорошо)
  - Если y=1, ŷ=0.1: loss ≈ -log(0.1) ≈ 2.30 (плохо)
  ```

- **metrics=['accuracy']**:
  - Отслеживаем метрику "точность" во время обучения
  - Accuracy = (правильно предсказано) / (всего)

### Строка 24: Вывод архитектуры
```python
model.summary()
```

**Выходной пример**:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 128)               1536
dropout (Dropout)            (None, 128)               0
dense_1 (Dense)              (None, 64)                8256
dropout_1 (Dropout)          (None, 64)                0
dense_2 (Dense)              (None, 32)                2080
dropout_2 (Dropout)          (None, 32)                0
dense_3 (Dense)              (None, 1)                 33
=================================================================
Total params: 11,905
Trainable params: 11,905
Non-trainable params: 0
```

---

## 🎓 Этап 6: Обучение модели

### Строки 1-18: Обучение нейросети
```python
history = model.fit(
    X_train_scaled, y_train,           # Входные данные и целевые значения
    batch_size=32,                      # Размер батча
    epochs=150,                         # Максимум эпох
    validation_split=0.2,               # 20% данных для валидации
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',     # Мониторим точность валидации
            patience=20,                # 20 эпох без улучшения = STOP
            restore_best_weights=True,  # Загрузить лучшие веса
            verbose=0                   # Не выводить сообщения
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',         # Мониторим потери валидации
            factor=0.5,                 # Умножить LR на 0.5
            patience=10,                # Если 10 эпох нет улучшения
            min_lr=1e-7,                # Минимальный LR
            verbose=0                   # Не выводить сообщения
        )
    ],
    verbose=1                           # Выводить прогресс (строка за эпоху)
)
```

**Детальное объяснение**:

#### Параметры fit()
- **X_train_scaled**: 5,197 × 11 матрица (5,197 образцов, 11 признаков)
- **y_train**: вектор из 5,197 целевых значений (0 или 1)
- **batch_size=32**: разбить 5,197 на батчи по 32 → ~163 батча
- **epochs=150**: повторять цикл максимум 150 раз
- **validation_split=0.2**: 20% от тренировочных данных использовать для валидации
  ```
  5,197 × 0.8 = 4,157 для обучения
  5,197 × 0.2 = 1,040 для валидации
  ```

#### Процесс одной эпохи
```
Эпоха 1:
  ├─ Батч 1: обновить веса на 32 образцах
  ├─ Батч 2: обновить веса на следующих 32 образцах
  ├─ ...
  ├─ Батч 163: обновить веса на последних образцах
  └─ Вычислить val_accuracy и val_loss на 1,040 валидационных образцах

Эпоха 2:
  ├─ Повторить батчи (в случайном порядке благодаря shuffle=True по умолчанию)
  └─ ...
```

#### EarlyStopping (ранняя остановка)
```
Назначение: Остановить обучение когда модель начинает переобучаться

Механизм:
  Эпоха 1:  val_acc = 0.820 → сохранить веса, счетчик = 0
  Эпоха 2:  val_acc = 0.850 ↑ → сохранить веса, счетчик = 0
  Эпоха 3:  val_acc = 0.870 ↑ → сохранить веса, счетчик = 0
  ...
  Эпоха 50: val_acc = 0.920 ↑ → сохранить веса, счетчик = 0 (ЛУЧШИЕ ВЕСА)
  Эпоха 51: val_acc = 0.918 ✗ → счетчик = 1 (без улучшения)
  Эпоха 52: val_acc = 0.915 ✗ → счетчик = 2
  ...
  Эпоха 70: val_acc = 0.915 ✗ → счетчик = 20
  → STOP! Обучение закончено

Загрузить веса из эпохи 50 (лучшие: 0.920)
```

**Параметры**:
- `monitor='val_accuracy'` - следим за точностью валидации
- `patience=20` - ждем 20 эпох без улучшения
- `restore_best_weights=True` - загрузить веса эпохи 50

#### ReduceLROnPlateau (уменьшение скорости обучения)
```
Назначение: Если модель "застряла", уменьшить learning rate

Механизм:
  Эпоха 1-10:   val_loss = 0.40 → 0.38 → 0.37 (улучшение)
  Эпоха 11-20:  val_loss = 0.37 (плато, нет улучшения)
  → Уменьшить lr с 0.001 на 0.0005
  Эпоха 21-30:  val_loss = 0.37 → 0.35 → 0.33 (улучшение продолжилось!)
  
Эффект: Маленький LR позволяет найти более точное минимум
```

**Параметры**:
- `monitor='val_loss'` - следим за потерями
- `factor=0.5` - умножить LR на 0.5 (новый = 0.001 × 0.5 = 0.0005)
- `patience=10` - после 10 эпох без улучшения
- `min_lr=1e-7` - не уменьшать LR ниже 1e-7

#### verbose=1 (вывод прогресса)
```
Эпоха 1:   1320/5197 [======>.......................] - 2s - loss: 0.4532 - accuracy: 0.7654 - val_loss: 0.3821 - val_accuracy: 0.8201
Эпоха 2:   1320/5197 [======>.......................] - 1s - loss: 0.3876 - accuracy: 0.8234 - val_loss: 0.3421 - val_accuracy: 0.8425
...

Интерпретация:
- 1320/5197: обработано 1320 из 5197 образцов
- loss: 0.4532: потеря на обучающем батче
- accuracy: 0.7654: точность на обучающем батче (76.54%)
- val_loss: 0.3821: потеря на валидационном наборе
- val_accuracy: 0.8201: точность на валидационном наборе (82.01%)
- 2s: примерно 2 секунды на эпоху
```

### Строка 19: Вывод истории обучения
```python
# Переменная history содержит:
# - history.history['accuracy']       → список точности по эпохам
# - history.history['val_accuracy']   → список точности валидации
# - history.history['loss']           → список потерь по эпохам
# - history.history['val_loss']       → список потерь валидации
```

---

## 📈 Этап 7: Оценка и визуализация

### Строки 1-15: Графики обучения
```python
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 строка, 2 столбца графиков

# ЛЕВЫЙ ГРАФИК: Accuracy
axes[0].plot(history.history['accuracy'], 
             label='Train Accuracy', linewidth=2.5, color='#667eea')
axes[0].plot(history.history['val_accuracy'], 
             label='Validation Accuracy', linewidth=2.5, color='#764ba2')
axes[0].set_title('Accuracy During Training', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0.4, 1.0])  # Ось Y от 0.4 до 1.0

# ПРАВЫЙ ГРАФИК: Loss
axes[1].plot(history.history['loss'], 
             label='Train Loss', linewidth=2.5, color='#f093fb')
axes[1].plot(history.history['val_loss'], 
             label='Validation Loss', linewidth=2.5, color='#4facfe')
axes[1].set_title('Loss During Training', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Интерпретация графиков**:

```
Идеальный сценарий:
  
  Accuracy                          Loss
  1.0  ╱────────────                0.5  ╲
       │                                  ╲────────
  0.9  │ ╱──────────                      ╲
       │╱          val (фиолет)      0.3   ╲
  0.8  ─────────── train (синий)           ╲
       │                            0.1     ╲───
       └────────────────────              ──────

Признаки переобучения:
  - Train acc: 0.95, Val acc: 0.82 (зазор > 0.1)
  - Val loss начинает расти (U-образная кривая)
  
Хороший результат:
  - Обе кривые (train и val) близко друг к другу
  - Обе кривые монотонно улучшаются
  - Ранняя остановка срабатывает вовремя
```

### Строки 16-31: Оценка на тестовом наборе
```python
# Получить предсказания
y_pred_proba = model.predict(X_test_scaled)  # Вероятности класса 1
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
# (вероятность > 0.5 → класс 1, иначе класс 0)

# Вычислить метрики
acc_nn = accuracy_score(y_test, y_pred)
roc_auc_nn = roc_auc_score(y_test, y_pred_proba)
```

**Процесс**:
```
X_test_scaled → [Нейросеть] → y_pred_proba
                                  ↓
                            Вероятности [0.1, 0.7, 0.3, ...]
                                  ↓
                            (> 0.5?) → [0, 1, 0, ...]
                                  ↓
                              y_pred
```

**Метрики**:
- **Accuracy**: (правильно) / (всего) = скільки процентів правильных предсказаний
- **ROC-AUC**: площадь под кривой ROC (0-1, где 1 = идеально)

### Строки 32-35: Классификационный отчет
```python
print(classification_report(y_test, y_pred, 
                          target_names=['Низкое', 'Высокое']))
```

**Выходной пример**:
```
              precision    recall  f1-score   support

      Низкое       0.95      0.97      0.96       780
      Высокое      0.94      0.88      0.91       520

    accuracy                           0.93      1300
   macro avg       0.94      0.93      0.93      1300
weighted avg       0.94      0.93      0.93      1300

Интерпретация:
- Precision (точность): из предсказанных "высоких", 94% реально высокие
- Recall (полнота): из реальных "высоких", модель нашла 88%
- F1-score: гармоническое среднее precision и recall
- Support: количество образцов каждого класса в тесте
```

### Строки 36-41: Матрица ошибок
```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Низкое', 'Высокое'],
            yticklabels=['Низкое', 'Высокое'])
plt.title('Матрица ошибок нейросети TensorFlow', fontsize=12, fontweight='bold')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.show()
```

**Матрица ошибок (пример)**:
```
                Pred: Низкое    Pred: Высокое
Real: Низкое        758                  22
Real: Высокое        43                 477

Интерпретация:
- TN (True Negative): 758 (низкое, предсказано низкое ✓)
- FP (False Positive): 22 (низкое, предсказано высокое ✗)
- FN (False Negative): 43 (высокое, предсказано низкое ✗)
- TP (True Positive): 477 (высокое, предсказано высокое ✓)

Accuracy = (TN + TP) / (всего) = (758 + 477) / 1300 = 0.95
```

---

## 🤖 Этап 8: Сравнение алгоритмов

### Строки 1-10: Определение 5 моделей
```python
models_dict = {
    'TensorFlow NN': None,                          # Уже обучена выше
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, 
        random_state=42, verbosity=0),
    'LightGBM': LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, 
        random_state=42, verbose=-1)
}
```

**Описание моделей**:

#### 1. Logistic Regression (Логистическая регрессия)
```
Самый простой метод:
  
  z = w₁×x₁ + w₂×x₂ + ... + w₁₁×x₁₁ + b
  ŷ = sigmoid(z) = 1 / (1 + e^(-z))
  
Преимущества:
- Очень быстро обучается
- Интерпретируемо (видно влияние каждого признака)
- Не требует много памяти
  
Ограничения:
- Не может моделировать нелинейные зависимости
- Часто менее точна чем другие методы

max_iter=1000: максимум 1000 итераций оптимизации
```

#### 2. Random Forest (Случайный лес)
```
Ансамбль из 200 деревьев решений:
  
  Дерево 1: IF алкоголь > 10.2 THEN высокое ELSE низкое
  Дерево 2: IF кислотность < 0.5 THEN высокое ELSE ...
  ...
  Дерево 200: ...
  
  Финальный результат: МАЖОРИТАРНОЕ ГОЛОСОВАНИЕ
  
Преимущества:
- Хорошо работает с табличными данными
- Невосприимчива к масштабированию
- Можно видеть важность признаков
  
Параметры:
- n_estimators=200: 200 деревьев
- max_depth=15: глубина каждого дерева ≤ 15
- n_jobs=-1: использовать все ядра процессора
```

#### 3. XGBoost (Extreme Gradient Boosting)
```
Последовательные деревья с коррекцией ошибок:
  
  Шаг 1: Обучить дерево 1, получить ошибки
  Шаг 2: Обучить дерево 2 на ошибках дерева 1
  Шаг 3: Обучить дерево 3 на ошибках деревьев 1+2
  ...
  Шаг 200: Финальный результат = сумма всех предсказаний
  
Преимущества:
- Часто лучше Random Forest
- Использует градиентный спуск (более умный поиск)
  
Параметры:
- n_estimators=200: 200 деревьев
- max_depth=6: каждое дерево неглубокое
- learning_rate=0.1: вес каждого нового дерева
```

#### 4. LightGBM (Light Gradient Boosting)
```
Быстрая версия XGBoost:
  
  Принцип: как XGBoost, но оптимизирован
  - Обрабатывает данные листьями (leaf-wise) вместо уровней
  - Меньше памяти
  - Быстрее обучается
  
Результат: часто сравнима с XGBoost по качеству, но на 2-10× быстрее
```

### Строки 11-25: Инициализация результатов и обучение
```python
results = {
    'TensorFlow NN': {
        'accuracy': acc_nn,
        'roc_auc': roc_auc_nn,
        'model': model
    }
}

print("\n🔄 Обучение других алгоритмов...\n")

for name, model_obj in list(models_dict.items())[1:]:
    print(f"  {name}...", end=' ')
    model_obj.fit(X_train_scaled, y_train)
    y_pred = model_obj.predict(X_test_scaled)
    y_pred_proba = model_obj.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    results[name] = {'accuracy': acc, 'roc_auc': roc_auc, 'model': model_obj}
    print(f"✓ Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")
```

**Процесс обучения**:
1. **Fit**: `model_obj.fit(X_train_scaled, y_train)` - обучение на 5,197 образцах
2. **Predict**: `model_obj.predict(X_test_scaled)` - предсказания на 1,300 образцах
3. **Probabilities**: `predict_proba()[:, 1]` - получить вероятности класса 1
4. **Evaluate**: вычислить accuracy и ROC-AUC
5. **Store**: сохранить результаты в словаре

**Временная сложность**:
```
Logistic Regression:  ~0.1s (очень быстро)
Random Forest:        ~1-2s
XGBoost:             ~2-3s
LightGBM:            ~1-2s
TensorFlow NN:       ~30-50s (с обучением, но используем историю)
```

---

## 📊 Этап 9: Финальные результаты

### Строки 1-13: Таблица сравнения
```python
comparison_df = pd.DataFrame({
    'Алгоритм': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'ROC-AUC': [results[name]['roc_auc'] for name in results.keys()]
})

comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print("\n" + comparison_df.to_string(index=False))
```

**Выходной пример**:
```
                  Алгоритм  Accuracy   ROC-AUC
0          TensorFlow NN      0.9342    0.9456
1               XGBoost      0.9223    0.9334
2              LightGBM      0.9169    0.9290
3          Random Forest      0.9154    0.9267
4  Logistic Regression      0.8523    0.9145
```

**Анализ**:
- TensorFlow NN лучший: 93.42% accuracy
- Разница с XGBoost: 1.2% (не очень много)
- Logistic Regression значительно хуже: 85.23%

### Строки 14-32: Визуализация сравнения
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ЛЕВЫЙ ГРАФИК: Сравнение Accuracy (горизонтальная гистограмма)
axes[0].barh(comparison_df['Алгоритм'], comparison_df['Accuracy'], 
             color='#667eea', alpha=0.8, edgecolor='black')
axes[0].set_title('Сравнение Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Accuracy')
axes[0].set_xlim([0.9, 1.0])  # Масштаб от 0.90 до 1.0 для лучшей видимости

# Добавить значения на столбцы
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0].text(v - 0.005, i, f'{v:.4f}', 
                ha='right', va='center', color='white', fontweight='bold')

# ПРАВЫЙ ГРАФИК: Сравнение ROC-AUC
axes[1].barh(comparison_df['Алгоритм'], comparison_df['ROC-AUC'], 
             color='#764ba2', alpha=0.8, edgecolor='black')
axes[1].set_title('Сравнение ROC-AUC', fontsize=12, fontweight='bold')
axes[1].set_xlabel('ROC-AUC')
axes[1].set_xlim([0.9, 1.0])

for i, v in enumerate(comparison_df['ROC-AUC']):
    axes[1].text(v - 0.005, i, f'{v:.4f}', 
                ha='right', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
```

**Описание графиков**:
- `barh()` - горизонтальная гистограмма (легче читать названия)
- `set_xlim([0.9, 1.0])` - масштаб от 0.90 до 1.0 (чтобы видны были разницы)
- Значения на столбцах помогают читать точные цифры

### Строки 33-49: ROC-кривые
```python
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#ff6b6b']

for idx, (name, result) in enumerate(results.items()):
    model_obj = result['model']
    
    # Получить вероятности предсказаний
    if name == 'TensorFlow NN':
        y_pred_proba = model.predict(X_test_scaled).flatten()
    else:
        y_pred_proba = model_obj.predict_proba(X_test_scaled)[:, 1]
    
    # Вычислить ROC-кривую
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = result['roc_auc']
    
    # Нарисовать
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', 
           linewidth=2.5, color=colors[idx])

# Линия случайного классификатора
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax.set_title('ROC Curves для всех моделей', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Объяснение ROC-кривой**:

```
ROC = Receiver Operating Characteristic

Оси:
  Y: True Positive Rate (TPR) = TP / (TP + FN)
     "Сколько положительных примеров найдено?"
  X: False Positive Rate (FPR) = FP / (FP + TN)
     "Сколько ложных срабатываний?"

Интерпретация:
  • (0, 0): классификатор говорит всегда "отрицательно" (никакие вина не высокого качества)
  • (1, 1): классификатор говорит всегда "положительно" (все вина высокого качества)
  • (0, 1): идеальный классификатор (находит всех, без ошибок)
  
AUC (Area Under the Curve):
  • 0.5: плохо (как случайное угадывание) - диагональная линия
  • 0.7-0.8: хорошо
  • 0.8-0.9: отличное
  • 0.9-1.0: выдающееся
  
В нашем случае:
  TensorFlow NN: AUC = 0.9456 (выдающееся)
  Logistic Regression: AUC = 0.9145 (отличное)
```

### Строки 50-53: Финальное сообщение
```python
print("\n" + "="*70)
print("✓ АНАЛИЗ ЗАВЕРШЁН!")
print("="*70)
print(f"\n🎯 ИТОГИ:")
print(f"  ✓ Точность нейросети TensorFlow: {acc_nn*100:.2f}%")
print(f"  ✓ ROC-AUC нейросети: {roc_auc_nn:.4f}")
print(f"  ✓ Лучший алгоритм: {comparison_df.iloc[0]['Алгоритм']}")
print(f"  ✓ Лучшая точность: {comparison_df.iloc[0]['Accuracy']*100:.2f}%")
print(f"  ✓ Использовано 5 алгоритмов для сравнения")
```

*Разработанная нейронная сеть представляет собой прочную основу для системы прогнозирования риска инсульта. Ключевые преимущества:
Модульность — четкое разделение алгоритмов
Масштабируемость — возможность добавления новых признаков и слоев

