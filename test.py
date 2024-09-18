import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests
import json
import pandas as pd
import time

def get_last_correct_spins():
    # URL of the JSON data
    url = "https://freeslotmania.com/stats.json"

    # Fetch the JSON data from the URL
    response = requests.get(url)
    data = response.json()

    # Conversion mapping for the result values
    result_conversion = {
        1: "one",
        2: "two",
        10: "ten",
        5: "five",
        400: "crazytime",
        300: "coinflip",
        200: "pachinko",
        100: "cashhunt"
    }

    # Extract the first 5 entries of result, multiplier, and total_payout
    spin_results = []
    for entry in data['data'][:1]:
        result = entry.get('result')
        result_word = result_conversion.get(result, "unknown")
        timestamp = entry.get('when') 
    return result_word, timestamp

# Step 1: Funzione per ottenere gli ultimi 5 spin dal sito web
def get_last_5_spins():
    # URL of the JSON data
    url = "https://freeslotmania.com/stats.json"

    # Fetch the JSON data from the URL
    response = requests.get(url)
    data = response.json()

    # Conversion mapping for the result values
    result_conversion = {
        1: "one",
        2: "two",
        10: "ten",
        5: "five",
        400: "crazytime",
        300: "coinflip",
        200: "pachinko",
        100: "cashhunt"
    }

    # Extract the first 5 entries of result, multiplier, and total_payout
    spin_results = []
    for entry in data['data'][:5]:
        result = entry.get('result')
        multiplier = entry.get('multiplier')
        total_payout = entry.get('total_payout')
        # Convert the result to its corresponding word
        result_word = result_conversion.get(result, "unknown")

        if multiplier is None:
            multiplier = entry.get('min_multiplier')
        if multiplier is None:
            multiplier =  entry.get('left_multiplier')  

            
        multiplier_with_x = f"{multiplier}X"

        spin_results.append({
            'multiplier': multiplier_with_x,
            'wheelResult': result_word,
            'totalPayout': int(total_payout)
        })
    return spin_results

# Step 2: Crea il modello LSTM
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(label_encoders['spinResultSymbol'].classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Carica e preprocessa i dati per codificare correttamente i valori
# Carica il file JSON locale per recuperare le classi conosciute
with open('crazy-time.json', 'r') as file:
    crazy_time_data = json.load(file)

# Converti i dati JSON in un DataFrame per codifica e normalizzazione
df = pd.DataFrame(crazy_time_data['data']['resultsData'])

# Seleziona le colonne rilevanti
df = df[['multiplier', 'spinResultSymbol', 'totalPayout']]  # Usa solo una volta 'spinResultSymbol'

# Codifica i dati categorici
label_encoders = {}
for column in ['multiplier', 'spinResultSymbol']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Funzione per validare il risultato della ruota
def is_valid_wheel_result(result):
    return '-' not in result  # Considera solo i risultati che non contengono '-'

# Step 4: Prepara i dati di input dagli spin
def prepare_input_from_spins(spins_list):
    inputs = []
    for spin in spins_list:
        if is_valid_wheel_result(spin['wheelResult']):  # Escludi i risultati complessi
            multiplier_encoded = label_encoders['multiplier'].transform([spin['multiplier']])[0]
            wheel_result_encoded = label_encoders['spinResultSymbol'].transform([spin['wheelResult']])[0]
            total_payout = spin['totalPayout']
            inputs.append([multiplier_encoded, wheel_result_encoded, total_payout])
    return np.array(inputs)

# Funzione per continuare ad addestrare il modello se viene punito
def retrain_model(model, input_data, target):
    # Ensure the input_data has the correct shape for the model (1, 3, 3)
    input_data = np.expand_dims(input_data, axis=0)  # Shape becomes (1, 3) -> (1, 1, 3)
    
    # Adjust input_data to match (batch_size, time_steps, features)
    input_data = np.expand_dims(input_data, axis=1)  # Shape becomes (1, 3, 3)

    model.fit(input_data, np.array([target]), epochs=1, verbose=0)
    print("Modello riaddestrato con l'ultimo risultato errato.")

    predict_and_compare_next_spin()  # Restart the prediction loop with the updated model


# Funzione principale per predire i risultati e confrontare con il nuovo risultato
def predict_and_compare_next_spin():
    # Recupera gli ultimi 5 spin dal sito
    spins_list = get_last_5_spins()

    # Prepara i dati di input per la predizione
    input_data = prepare_input_from_spins(spins_list)

    if len(input_data) == 0:
        print("Nessun spin valido disponibile per la predizione.")
        return

    # Normalizza i dati di input
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Aggiungi una dimensione per il batch, in modo da avere (1, n, 3) come input
    input_data_scaled = np.expand_dims(input_data_scaled, axis=0)

    # Step 5: Predici il prossimo spin usando i dati sequenziali
    predicted_spin = model.predict(input_data_scaled)

    # Decodifica la previsione
    predicted_class_index = np.argmax(predicted_spin, axis=1)[0]
    predicted_wheel_result = label_encoders['spinResultSymbol'].inverse_transform([predicted_class_index])[0]

    # Stampa il risultato previsto
    print(f"Ultimi 5:")
    for spin in spins_list:
        print(spin["wheelResult"])



    # Step 6: Attendi un nuovo risultato nel JSON
    last_spin, last_timestamp = get_last_correct_spins()  # Memorizza l'ultimo risultato per il confronto
    print(f"Ultimo risultato attuale: {last_spin} In attesa di un nuovo risultato...")
    print(f"RISULTATO PREVISTO: {predicted_wheel_result}")

    while True:
        # Controlla ogni 2 secondi per un nuovo risultato
        time.sleep(2)
        correct_spin_last, correct_timestamp = get_last_correct_spins()

        # Confronta il nuovo spin con l'ultimo spin memorizzato
        if correct_spin_last != last_spin or correct_timestamp != last_timestamp:
            actual_result = correct_spin_last
            print(f"Nuovo risultato trovato: {actual_result}")
            break

    # Step 7: Premia o punisci l'AI in base alla corrispondenza
    if predicted_wheel_result == actual_result:
        print("Risultato corretto! AI premiata.")
        predict_and_compare_next_spin()  # Restart the prediction loop with the updated model

    else:
        print("Risultato errato! AI punita.")
        # Riaddestra il modello usando l'ultimo risultato effettivo
        actual_class_index = label_encoders['spinResultSymbol'].transform([actual_result])[0]
        retrain_model(model, input_data[-1], actual_class_index)

# Step 8: Avvia la predizione e attesa di nuovi valori dopo la prima predizione
input_shape = (df.shape[1], 3)  # Dimensioni di input corrette per il modello
model = create_lstm_model(input_shape)

# Esegui la predizione e avvia il ciclo di controllo di nuovi valori
predict_and_compare_next_spin()
