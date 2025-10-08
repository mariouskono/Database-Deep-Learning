import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# =====================
# 1. Load Data dari SQLite
# =====================
conn = sqlite3.connect("iris.db")
query = "SELECT SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species FROM iris"
df = pd.read_sql_query(query, conn)
conn.close()

print("Preview dataset:")
print(df.head())

# =====================
# 2. Preprocessing
# =====================
# Encode label Species
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# Pisahkan fitur dan target
X = df.drop("Species", axis=1).values
y = df["Species"].values

# Normalisasi
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# 3. Build Model Deep Learning
# =====================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 kelas iris
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# 4. Training
# =====================
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_split=0.2,
    verbose=1
)

# =====================
# 5. Evaluasi
# =====================
loss, acc = model.evaluate(X_test, y_test)
print(f"Akurasi model: {acc:.2f}")

# Prediksi
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Laporan klasifikasi
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Confusion matrix
print(confusion_matrix(y_test, y_pred))