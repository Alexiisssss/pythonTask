
# В домашней работе необходимо с помощью AutoKeras или KerasTuner найти оптимальную модель для решения одной из следующей задач:
# Используйте русский корпус новостей от Lenta.ru подберите и обучите модель классифицировать новости по заголовкам на классы (поле topic в датасете).
# Используйте 9 самых часто встречаемых топиков и 10-й для остальных, не вошедших в 9 классов.
# Оцените модель с помощью отчета о классификации, сделайте выводы.


import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv("dataset/lenta-ru-news.csv", low_memory=False)
print(df.head())

# Оставляем только столбцы title и topic
df = df[['title', 'topic']]

# Выбираем 9 самых частых топиков
top_topics = df['topic'].value_counts().nlargest(9).index
df['topic'] = df['topic'].apply(lambda x: x if x in top_topics else 'other')

# Препроцессинг данных
X = df['title'].astype(str)
y = df['topic'].astype(str)

# Кодирование целевых меток
y_encoded = pd.get_dummies(y).values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Создание текстовой векторизации
text_vectorization = layers.TextVectorization(max_tokens=20000, output_sequence_length=250)
text_vectorization.adapt(X_train)

# Определение функции для построения модели
def build_model():
    model = tf.keras.Sequential()
    model.add(text_vectorization)
    model.add(layers.Embedding(input_dim=20000, output_dim=256))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(y_encoded.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Построение и обучение модели
model = build_model()
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Оценка модели
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Отчет о классификации
target_names = df['topic'].unique()
report = classification_report(y_true_classes, y_pred_classes, target_names=target_names)
print("Classification Report:\n", report)

# Построение и отображение матрицы ошибок
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

