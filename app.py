import tkinter as tk
from tkinter import messagebox
import spacy
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar spaCy
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

# Cargar tokenizer y modelo entrenado
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model1 = load_model("model1.h5")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
    
with open("model2.pkl", "rb") as f:
    model2 = pickle.load(f)

maxlen = 150

# Diccionario de clases (ejemplo)
clases_polaridad = ["Muy Negativa", "Negativa", "Neutral", "Positiva", "Muy Positiva"]
clases_tipo = ["Restaurante", "Atracción", "Hotel"]

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Texto")
        self.root.geometry("400x300")

        self.text_input = tk.Text(self.root, height=6, width=40)
        self.text_input.pack(pady=10)

        self.boton_predecir = tk.Button(self.root, text="Predecir", command=self.predecir)
        self.boton_predecir.pack(pady=5)

        self.label_polaridad = tk.Label(self.root, text="Polaridad: ")
        self.label_polaridad.pack(pady=5)

        self.label_tipo = tk.Label(self.root, text="Tipo: ")
        self.label_tipo.pack(pady=5)

    def predecir(self):
        texto = self.text_input.get("1.0", tk.END).strip()
        if texto:
            # Normalización y lematización
            doc = nlp(texto.lower())
            lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
            texto_procesado = " ".join(lemmas)

            # Tokenización y padding
            secuencia = tokenizer.texts_to_sequences([texto_procesado])
            secuencia_padded = pad_sequences(secuencia, maxlen=maxlen)

            # Predicción
            pred1 = model1.predict(secuencia_padded)[0]
            clase_pred1 = np.argmax(pred1)
            
            clase_pred2 = int(model2.predict(vectorizer.transform([texto_procesado]))[0])-1

            self.label_polaridad.config(text=f"Polaridad: {clases_polaridad[clase_pred1]}")
            self.label_tipo.config(text=f"Tipo: {clases_tipo[clase_pred2]}")

        else:
            messagebox.showwarning("Advertencia", "Por favor ingresa un texto.")

# Ejecutar la app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()




