

# ============================
# 0. Импорт
# ============================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)

# ============================
# 1. Данные IMDB (урезаем для скорости)
# ============================
vocab_size = 10000   # меньше слов -> быстрее [web:29]
max_len = 100        # короче последовательность -> быстрее

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# Берём только часть данных, чтобы обучать быстрее
n_train = 15000
n_test = 5000
x_train, y_train = x_train[:n_train], y_train[:n_train]
x_test, y_test = x_test[:n_test], y_test[:n_test]

x_train = keras.utils.pad_sequences(x_train, maxlen=max_len, padding="post", truncating="post")
x_test = keras.utils.pad_sequences(x_test, maxlen=max_len, padding="post", truncating="post")

print("Train shape:", x_train.shape, "Test shape:", x_test.shape)

# ============================
# 2. Positional Encoding
# ============================
class PositionalEncoding(layers.Layer):
    def __init__(self, max_position=5000, embedding_dim=128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_position = max_position
        self.embedding_dim = embedding_dim

        position = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, embedding_dim, 2) * (-np.log(10000.0) / embedding_dim)
        )

        pe = np.zeros((max_position, embedding_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.cast(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

# ============================
# 3. Multi-Head Self-Attention
# ============================
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embedding_dim, num_heads, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.depth = embedding_dim // num_heads

        self.Wq = layers.Dense(embedding_dim)
        self.Wk = layers.Dense(embedding_dim)
        self.Wv = layers.Dense(embedding_dim)
        self.dense = layers.Dense(embedding_dim)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = self._split_heads(Q, batch_size)
        K = self._split_heads(K, batch_size)
        V = self._split_heads(V, batch_size)

        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
        output = tf.matmul(attention_weights, V)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.embedding_dim))
        output = self.dense(concat_attention)
        return output

# ============================
# 4. FeedForward и EncoderLayer
# ============================
class FeedForward(layers.Layer):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate=0.1, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dense2 = layers.Dense(embedding_dim)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        x_ff = self.dense1(x)
        x_ff = self.dropout(x_ff, training=training)
        x_ff = self.dense2(x_ff)
        return x_ff

class EncoderLayer(layers.Layer):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.ffn = FeedForward(embedding_dim, hidden_dim, dropout_rate)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        attn_output = self.mha(x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# ============================
# 5. Transformer-классификатор (упрощённый)
# ============================
embedding_dim = 64     # поменьше для скорости
num_heads = 4
hidden_dim = 128
num_layers = 1         # один EncoderLayer -> быстрее
dropout_rate = 0.1

inputs = keras.Input(shape=(max_len,), dtype="int32")

x = layers.Embedding(vocab_size, embedding_dim)(inputs)
x = PositionalEncoding(max_position=max_len, embedding_dim=embedding_dim)(x)

for _ in range(num_layers):
    x = EncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# ============================
# 6. Компиляция и обучение
# ============================
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)

epochs = 5        # мало эпох -> быстро, но ~0.7 достижимо [web:24][web:28]
batch_size = 128

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# ============================
# 7. Оценка на тесте
# ============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# ============================
# 8. Графики метрик (accuracy и loss)
# ============================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================
# 9. Примеры предсказаний
# ============================
word_index = keras.datasets.imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
index_word[0] = "<pad>"
index_word[1] = "<start>"
index_word[2] = "<unk>"

def decode_review(encoded_review):
    return " ".join(index_word.get(i, "?") for i in encoded_review)

for i in range(3):
    sample = x_test[i:i+1]
    pred = model.predict(sample)[0][0]
    label = y_test[i]
    print("=" * 80)
    print("Текст отзыва:\n", decode_review(sample[0]))
    print("Истинный класс:", "positive" if label == 1 else "negative")
    print("Предсказанный score:", float(pred), "->", "positive" if pred >= 0.5 else "negative")
