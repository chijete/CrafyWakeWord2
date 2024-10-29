import librosa
import pandas as pd
from datasets import load_dataset, Dataset
import json
import sys
import numpy as np
import os
import shutil
from pydub import AudioSegment

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

# Función para cargar audio y procesarlo
def load_audio(file, target_length):
  target_length = int(target_length)

  audio = AudioSegment.from_file(file)

  current_db = audio.dBFS
  gain = -23 - current_db
  audio = audio.apply_gain(gain)

  audio = audio.set_frame_rate(16000)  # Remuestrear a 16 kHz
  audio = audio.set_channels(1)  # Convertir a mono
  samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
  samples = samples / np.max(np.abs(samples))  # Normalizar entre -1 y 1

  if len(samples) < target_length:
    pad_width = target_length - len(samples)
    samples = np.pad(samples, (0, int(pad_width)), mode='constant')
  else:
    samples = samples[:target_length]
  
  audio.export('test_audio_l.wav', format="wav")
  
  return samples

fsamples = load_audio('test_audio.wav', 16000*(config_datos['max_audio_length'] / 1000))

print(len(fsamples.tolist()))
with open('test_audio_2.json', 'w') as f:
  json.dump(fsamples.tolist(), f)