import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# 1. Koneksi MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["imdb_db"]
col = db["reviews"]

# 2. Import CSV ke MongoDB
df_csv = pd.read_csv("IMDB Dataset.csv") 
# Kolom: “review”, “sentiment”
records = df_csv.to_dict(orient="records")
col.insert_many(records)

# 3. Ambil data dari MongoDB
docs = list(col.find({}, {"_id": 0}))
df = pd.DataFrame(docs)
print(df.head())


# 4. Preprocessing label
encoder = LabelEncoder()
df["sentiment_label"] = encoder.fit_transform(df["sentiment"])  # e.g. negative=0, positive=1

# 5. Preprocessing teks
texts = df["review"].values
labels = df["sentiment_label"].values

# Tokenisasi & index
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
seq = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(seq, maxlen=200, padding="post", truncating="post")

# Split training & testing
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42, stratify=labels
)

# 6. Bangun model sederhana (Embedding + LSTM)
model = Sequential([
    layers.Embedding(input_dim=10000, output_dim=64, input_length=200),
    layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Training dengan callback
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    batch_size=32,
    callbacks=[es, rlr],
    verbose=1
)

# Evaluasi
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print(confusion_matrix(y_test, y_pred))
