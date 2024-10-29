import onnxruntime as ort
import numpy as np
import librosa
import json
import sys
from pydub import AudioSegment
import time
import random
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Model test ONNX')
parser.add_argument('-model', type=str, required=False, help='Filename of ONNX model to optimize')
args = parser.parse_args()

# Cargar la configuración
with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

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

# Configuración del modelo ONNX
model_path_name = "model.onnx"
if args.model:
  model_path_name = args.model

onnx_model_path = './model/'+model_path_name
session = ort.InferenceSession(onnx_model_path)

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
def predict(session, audio_file_path):
  audio = load_audio(audio_file_path, 16000*(config_datos['max_audio_length'] / 1000))
  # Convertir a tensor numpy
  input_array = np.expand_dims(audio, axis=0).astype(np.float32)
  # Realizar inferencia
  inputs = {session.get_inputs()[0].name: input_array}
  logits = session.run(None, inputs)[0]
  predicted_class_id = np.argmax(logits, axis=-1).item()
  return predicted_class_id

# Evaluar en un conjunto de pruebas
print('Testing the model!')
test_files = files_test
wins = 0
wins_positive = 0
wins_negative = 0
all_inference_times = []
forIndex = 0
for file in test_files:
  init_time = time.time()
  predicted_class = predict(session, file)
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
  print(f'File: {file}, Must be: {mustClass}, Predicted class: {predicted_class}, In: {total_time_str} s')
  forIndex = forIndex + 1

average_inference_time = sum(all_inference_times) / len(all_inference_times)

wins_percent = wins * 100 / len(test_files)
print('Wins: '+str(wins)+' of '+str(len(test_files))+' = '+str(wins_percent)+'%')
print('Wins positive: '+str(wins_positive)+' of '+str(df_test_positive.shape[0]))
print('Wins negative: '+str(wins_negative)+' of '+str(df_test_negative.shape[0]))
print('Average inference time in seconds: '+str(average_inference_time))
print('Model: '+onnx_model_path)

with open("./model/"+model_path_name+"_test.json", 'w') as file:
  json.dump({
    "accuracy": wins_percent,
    "correct_positive_items": wins_positive,
    "total_positive_items": df_test_positive.shape[0],
    "correct_negative_items": wins_negative,
    "total_negative_items": df_test_negative.shape[0],
    "average_inference_time": average_inference_time
  }, file, indent=4)

sys.exit()

test_files = [
  "test_true.wav",
  "test_false.wav"
]
for file in test_files:
  init_time = time.time()
  predicted_class = predict(session, file)
  total_time = str(time.time() - init_time)
  print(f'File: {file}, Predicted class: {predicted_class}, In: {total_time} ms')
