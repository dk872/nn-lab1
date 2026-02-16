import tensorflow as tf
import numpy as np

# Дані XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

# Фіксуємо seed
tf.random.set_seed(42)
np.random.seed(42)

# Модель
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компіляція
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy'
)

# Навчання
model.fit(X, y, epochs=1500, verbose=0)

# Перевірка
predictions = model.predict(X)

print("Вхід -> Передбачення")
for inp, pred in zip(X, predictions):
    print(f"{inp} -> {pred[0]:.4f}")
