
# ==========================================
# ПОЛНЫЙ КОД ДЛЯ GOOGLE COLAB
# Wine Quality Prediction with TensorFlow
# ==========================================

# --- 1. УСТАНОВКА БИБЛИОТЕК ---
print("📦 Установка библиотек...")
!pip install -q tensorflow scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm

# --- 2. ИМПОРТ ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import warnings
warnings.filterwarnings('ignore')

# Настройка графиков
plt.rcParams['figure.figsize'] = (14, 8)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# --- 3. ЗАГРУЗКА ДАТАСЕТА ---
print("\n" + "="*70)
print("📥 ЗАГРУЗКА ДАТАСЕТА")
print("="*70)

url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

print("\n✓ Загрузка красного вина...")
df_red = pd.read_csv(url_red, sep=';')
df_red['wine_type'] = 'red'

print("✓ Загрузка белого вина...")
df_white = pd.read_csv(url_white, sep=';')
df_white['wine_type'] = 'white'

# Объединение
df = pd.concat([df_red, df_white], ignore_index=True)

print(f"\n✓ Датасет загружен!")
print(f"  Форма: {df.shape}")
print(f"  Красного вина: {len(df_red)}")
print(f"  Белого вина: {len(df_white)}")

# --- 4. АНАЛИЗ ДАННЫХ ---
print("\n" + "="*70)
print("📊 АНАЛИЗ ДАННЫХ")
print("="*70)

print("\nРаспределение качества:")
print(df['quality'].value_counts().sort_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['quality'], bins=20, color='#667eea', alpha=0.7, edgecolor='black')
axes[0].set_title('Распределение качества вина', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Качество')
axes[0].set_ylabel('Количество')

df_red['quality'].hist(bins=15, alpha=0.6, label='Red', ax=axes[1], color='#e74c3c')
df_white['quality'].hist(bins=15, alpha=0.6, label='White', ax=axes[1], color='#f1c40f')
axes[1].set_title('Распределение по типам вина', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

# --- 5. ПРЕДОБРАБОТКА ---
print("\n" + "="*70)
print("🔄 ПРЕДОБРАБОТКА ДАННЫХ")
print("="*70)

df = df.drop_duplicates().reset_index(drop=True)
print(f"✓ Дубликаты удалены")

# Бинарная классификация
df['quality_binary'] = (df['quality'] > 6).astype(int)
print(f"\nРаспределение классов:")
print(df['quality_binary'].value_counts())

# Выделение признаков
X = df.drop(columns=['quality', 'quality_binary', 'wine_type'])
y = df['quality_binary']

print(f"\n✓ Признаки (11 химических показателей):")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# Разделение и нормализация
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Выборки:")
print(f"  Обучающая: {X_train_scaled.shape}")
print(f"  Тестовая: {X_test_scaled.shape}")

# --- 6. ПОЛНОСВЯЗНАЯ НЕЙРОСЕТЬ НА TENSORFLOW ---
print("\n" + "="*70)
print("🧠 ПОСТРОЕНИЕ НЕЙРОННОЙ СЕТИ TENSORFLOW")
print("="*70)

print("\n📋 Параметры нейросети:")
print("  input_dim: 11")
print("  hidden_layers: [128, 64, 32]")
print("  activation: relu")
print("  dropout_rate: 0.3")
print("  l2_reg: 1e-4")
print("  learning_rate: 0.001")
print("  batch_size: 32")
print("  epochs: 150")


# Построение модели
model = keras.Sequential([
    layers.Input(shape=(11,)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='hidden_1'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='hidden_2'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='hidden_3'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Архитектура:")
model.summary()

# Обучение
print("\n🔄 Обучение (150 эпох)...")
history = model.fit(
    X_train_scaled, y_train,
    batch_size=32,
    epochs=150,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)
    ],
    verbose=1
)

# --- 7. ГРАФИКИ ОБУЧЕНИЯ ---
print("\n" + "="*70)
print("📈 ГРАФИКИ ОБУЧЕНИЯ")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2.5, color='#667eea')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2.5, color='#764ba2')
axes[0].set_title('Accuracy During Training', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0.4, 1.0])

axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2.5, color='#f093fb')
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5, color='#4facfe')
axes[1].set_title('Loss During Training', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --- 8. ОЦЕНКА НЕЙРОСЕТИ ---
print("\n" + "="*70)
print("🎯 ОЦЕНКА НЕЙРОСЕТИ")
print("="*70)

y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

acc_nn = accuracy_score(y_test, y_pred)
roc_auc_nn = roc_auc_score(y_test, y_pred_proba)

print(f"\n✓ Метрики:")
print(f"  Accuracy: {acc_nn:.4f} ({acc_nn*100:.2f}%)")
print(f"  ROC-AUC: {roc_auc_nn:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Низкое', 'Высокое']))

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

# --- 9. СРАВНЕНИЕ С ДРУГИМИ АЛГОРИТМАМИ ---
print("\n" + "="*70)
print("📊 СРАВНЕНИЕ С ДРУГИМИ АЛГОРИТМАМИ (5 АЛГОРИТМОВ)")
print("="*70)

models_dict = {
    'TensorFlow NN': None,
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
    'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
}

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

# Таблица результатов
print("\n" + "="*70)
print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*70)

comparison_df = pd.DataFrame({
    'Алгоритм': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'ROC-AUC': [results[name]['roc_auc'] for name in results.keys()]
})

comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print("\n" + comparison_df.to_string(index=False))

# Визуализация сравнения
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(comparison_df['Алгоритм'], comparison_df['Accuracy'], color='#667eea', alpha=0.8, edgecolor='black')
axes[0].set_title('Сравнение Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Accuracy')
axes[0].set_xlim([0.9, 1.0])
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0].text(v - 0.005, i, f'{v:.4f}', ha='right', va='center', color='white', fontweight='bold')

axes[1].barh(comparison_df['Алгоритм'], comparison_df['ROC-AUC'], color='#764ba2', alpha=0.8, edgecolor='black')
axes[1].set_title('Сравнение ROC-AUC', fontsize=12, fontweight='bold')
axes[1].set_xlabel('ROC-AUC')
axes[1].set_xlim([0.9, 1.0])
for i, v in enumerate(comparison_df['ROC-AUC']):
    axes[1].text(v - 0.005, i, f'{v:.4f}', ha='right', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# ROC Curves
print("\n🔄 Построение ROC-кривых...\n")
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#ff6b6b']

for idx, (name, result) in enumerate(results.items()):
    model_obj = result['model']
    if name == 'TensorFlow NN':
        y_pred_proba = model.predict(X_test_scaled).flatten()
    else:
        y_pred_proba = model_obj.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = result['roc_auc']
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', linewidth=2.5, color=colors[idx])

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax.set_title('ROC Curves для всех моделей', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("✓ АНАЛИЗ ЗАВЕРШЁН!")
print("="*70)
print(f"\n🎯 ИТОГИ:")
print(f"  ✓ Точность нейросети TensorFlow: {acc_nn*100:.2f}%")
print(f"  ✓ ROC-AUC нейросети: {roc_auc_nn:.4f}")
print(f"  ✓ Лучший алгоритм: {comparison_df.iloc[0]['Алгоритм']}")
print(f"  ✓ Лучшая точность: {comparison_df.iloc[0]['Accuracy']*100:.2f}%")
print(f"  ✓ Использовано 5 алгоритмов для сравнения")
