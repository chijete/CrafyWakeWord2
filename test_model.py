from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, HubertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import librosa
import pandas as pd
from datasets import load_dataset, Dataset
import json
import sys
import numpy as np
from pydub import AudioSegment
import random
import time

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

model_path_name = "model"
model_path = "./model/"+model_path_name
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
if config_datos['base_model_type'] == 'hubert':
  model = HubertForSequenceClassification.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True)
else:
  model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(config_datos['processor_path'])

# Función para cargar audio y procesarlo
def load_audio(file, target_length):
  target_length = int(target_length)

  audio = AudioSegment.from_file(file)

  if config_datos['volume_normalization']:
    current_db = audio.dBFS
    gain = -23 - current_db
    audio = audio.apply_gain(gain)

  # if config_datos['volume_randomize']:
  #   audio = audio.apply_gain(random.randint(config_datos['volume_randomize_limits'][0], config_datos['volume_randomize_limits'][1]))

  audio = audio.set_frame_rate(16000)  # Remuestrear a 16 kHz
  audio = audio.set_channels(1)  # Convertir a mono
  samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
  samples = samples / np.max(np.abs(samples))  # Normalizar entre -1 y 1

  if len(samples) < target_length:
    pad_width = target_length - len(samples)
    samples = np.pad(samples, (0, int(pad_width)), mode='constant')
  else:
    samples = samples[:target_length]
  
  return samples

# Función para realizar la inferencia
def predict(model, processor, audio_file_path):
  audio = load_audio(audio_file_path, 16000*(config_datos['max_audio_length'] / 1000))
  inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
  with torch.no_grad():
    logits = model(**inputs).logits
  predicted_class_id = torch.argmax(logits, dim=-1).item()
  return predicted_class_id

# Evaluar en un conjunto de pruebas
print('Testing the model!')
test_files = files_test
# test_files = [
#   "test_true.wav",
#   "test_false.wav",
#   "test_audio.wav",
#   "test_audio_2.wav",
# ]
wins = 0
wins_positive = 0
wins_negative = 0
all_inference_times = []
forIndex = 0
for file in test_files:
  init_time = time.time()
  predicted_class = predict(model, processor, file)
  total_time = time.time() - init_time
  total_time_str = str(total_time)
  all_inference_times.append(total_time)
  mustClass = str(labels_test[forIndex])
  if str(mustClass) == str(predicted_class):
    wins = wins + 1
    if 'positive/' in file:
      wins_positive = wins_positive + 1
    elif 'negative/' in file:
      wins_negative = wins_negative + 1
  print(f'File: {file}, Must be: {mustClass}, Predicted class: {predicted_class}')
  forIndex = forIndex + 1

average_inference_time = sum(all_inference_times) / len(all_inference_times)

wins_percent = wins * 100 / len(test_files)
print('Wins: '+str(wins)+' of '+str(len(test_files))+' = '+str(wins_percent)+'%')
print('Wins positive: '+str(wins_positive)+' of '+str(df_test_positive.shape[0]))
print('Wins negative: '+str(wins_negative)+' of '+str(df_test_negative.shape[0]))
print('Average inference time in seconds: '+str(average_inference_time))
print('Model: '+model_path)

with open("./model/"+model_path_name+"_test.json", 'w') as file:
  json.dump({
    "accuracy": wins_percent,
    "correct_positive_items": wins_positive,
    "total_positive_items": df_test_positive.shape[0],
    "correct_negative_items": wins_negative,
    "total_negative_items": df_test_negative.shape[0],
    "average_inference_time": average_inference_time
  }, file, indent=4)