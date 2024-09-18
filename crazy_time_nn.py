import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Step 1: Carica il file JSON
with open('crazy-time.json', 'r') as file:
    crazy_time_data = json.load(file)

# Step 2: Converti i dati JSON in un DataFrame
df = pd.DataFrame(crazy_time_data['data']['resultsData'])

# Step 3: Seleziona le colonne rilevanti per l'addestramento
# Usiamo i moltiplicatori e payout come feature per predire il 'spinResultSymbol'
df = df[['multiplier', 'slotResultSymbol', 'spinResultSymbol', 'totalPayout']]

# Step 4: Codifica i dati categorici
label_encoders = {}
for column in ['multiplier', 'slotResultSymbol', 'spinResultSymbol']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 5: Definisci X (caratteristiche) e y (target)
X = df[['multiplier', 'slotResultSymbol', 'totalPayout']]
y = df['spinResultSymbol']

# Step 6: Dividi i dati in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Normalizza i dati numerici
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Crea il modello di rete neurale
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(label_encoders['spinResultSymbol'].classes_), activation='softmax')  # Output layer for classification
])

# Step 9: Compila il modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 10: Addestra il modello
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 11: Valuta il modello sui dati di test
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Step 12: Salva il modello
model.save('crazy_time_spin_model.h5')

# Step 13: Visualizza la storia dell'addestramento
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
