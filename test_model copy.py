from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import librosa
import pandas as pd
from datasets import load_dataset, Dataset
import json
import sys
import numpy as np

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

model_path = "./model/model"
processor_path = "./model/processor"

# Test dataset

dataset_test_positive = pd.read_csv('datasets/test/positive/dataset.csv')
dataset_test_negative = pd.read_csv('datasets/test/negative/dataset.csv')

df_test_positive = dataset_test_positive[['path']]
df_test_positive['label'] = 1
df_test_positive = df_test_positive.rename(columns={'path': 'file_path'})

df_test_negative = dataset_test_negative[['path']]
df_test_negative['label'] = 0
df_test_negative = df_test_negative.rename(columns={'path': 'file_path'})

df_test = pd.concat([df_test_positive, df_test_negative], ignore_index=True)
files_test = df_test['file_path'].tolist()
labels_test = df_test['label'].tolist()

# Cargar el modelo y el procesador
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(processor_path)

# Función para cargar audio y procesarlo
def load_audio(file, target_length):
  target_length = int(target_length)
  audio, _ = librosa.load(file, sr=16000)
  if len(audio) < target_length:
    pad_width = target_length - len(audio)
    audio = np.pad(audio, (0, int(pad_width)), mode='constant')
  else:
    audio = audio[:target_length]
  return audio

# Función para realizar la inferencia
def predict(model, processor, audio_file_path):
  audio = load_audio(audio_file_path, 16000*(config_datos['max_audio_length'] / 1000))
  inputs = processor(audio, return_tensors="pt", sampling_rate=16000, padding='max_length', max_length=48000, truncation=True)
  with torch.no_grad():
    logits = model(**inputs).logits
  predicted_class_id = torch.argmax(logits, dim=-1).item()
  return predicted_class_id

# Evaluar en un conjunto de pruebas
print('Testing the model!')
test_files = files_test
test_files = [
  "test_true.wav",
  "test_false.wav"
]
for file in test_files:
  predicted_class = predict(model, processor, file)
  print(f'File: {file}, Predicted class: {predicted_class}')