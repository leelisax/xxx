***RNN для генерации текста на основе обучающего набора.***
**Задание 7: RNN для генерации текста**
Задача: создать рекуррентную нейронную сеть (RNN) для генерации текста на основе обучающего набора. 
 
Требования: 
Использовать LSTM ячейки (128 units) 
Embedding layer для векторизации символов 
Предсказание следующего символа 
Управление температурой при генерации 
 
Код-заготовка (Python): 
 
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
 
# Что нужно дополнить: 
# 1. Слои Embedding и LSTM 
# 2. Функцию preprocess_text с character-level encoding 
# 3. Обучение модели с callback для мониторинга 
# 4. Реализацию temperature-based sampling 
# 5. Экспериментирование с разными значениями temperature 
