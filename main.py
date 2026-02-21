import tensorflow as tf
import numpy as np
import itertools

# Генерація таблиці істинності для 4 змінних (16 рядків)
X = np.array(list(itertools.product([0, 1], repeat=4)), dtype=np.float32)

# 2. Обчислення Y для 4-х змінних
# Якщо кількість одиниць непарна -> 1, інакше -> 0
y = np.array([[np.sum(x) % 2] for x in X], dtype=np.float32)

# Фіксуємо seed
tf.random.set_seed(42)
np.random.seed(42)

# Модель
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компіляція
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy', metrics=['accuracy'])

# Навчання
model.fit(X, y, epochs=1500, verbose=0)

# Перевірка
predictions = model.predict(X)
print("Вхід -> Передбачення")
for inp, pred in zip(X, predictions):
    print(f"{inp} -> {pred[0]:.4f} ({int(np.round(pred[0]))})")
