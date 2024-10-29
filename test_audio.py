from pydub import AudioSegment
import numpy as np
import json
import sys
import librosa

target_length = 16000*(1800 / 1000)
target_length = int(target_length)

def process_audio(file_path):
  audio = AudioSegment.from_file(file_path)
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

def process_audio_librosa(file_path):
  audio, _ = librosa.load(file_path, sr=16000)
  samples = np.array(audio, dtype=np.float32)
  samples = samples / np.max(np.abs(samples))  # Normalizar entre -1 y 1

  if len(samples) < target_length:
    pad_width = target_length - len(samples)
    samples = np.pad(samples, (0, int(pad_width)), mode='constant')
  else:
    samples = samples[:target_length]
  
  return samples

audio_path = 'test_audio_2.wav'
samples = process_audio(audio_path)
samples_librosa = process_audio_librosa(audio_path)

print(len(samples.tolist()))
with open('test_audio.json', 'w') as f:
  json.dump(samples.tolist(), f)

with open('test_audio_librosa.json', 'w') as f:
  json.dump(samples_librosa.tolist(), f)

def compare_json_files(file1_path, file2_path):
  # Leer y cargar los archivos JSON
  with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
      json_data1 = json.load(file1)
      json_data2 = json.load(file2)
  
  # Comparar los datos JSON
  return json_data1 == json_data2

file1_path = 'test_audio.json'
file2_path = 'test_audio_librosa.json'

if compare_json_files(file1_path, file2_path):
  print("Los archivos JSON son idénticos.")
else:
  print("Los archivos JSON no son idénticos.")