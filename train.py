from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, HubertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import librosa
import pandas as pd
from datasets import load_dataset, Dataset
import json
import sys
import numpy as np
import os
import shutil
from pydub import AudioSegment
import random
import time

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

pretrained_model = config_datos['base_model']

if os.path.isdir('model/'):
  shutil.rmtree('model/')

os.mkdir('model/')

file_init_time = time.time()

# Cargar y preprocesar tu dataset
# Supongamos que tienes un DataFrame con columnas 'file_path' y 'label'

dataset_train_positive = pd.read_csv('datasets/train/positive/dataset.csv')
dataset_train_negative = pd.read_csv('datasets/train/negative/dataset.csv')
dataset_test_positive = pd.read_csv('datasets/test/positive/dataset.csv')
dataset_test_negative = pd.read_csv('datasets/test/negative/dataset.csv')

# Train dataset

df_train_positive = dataset_train_positive[['path']]
df_train_positive['label'] = 1
df_train_positive = df_train_positive.rename(columns={'path': 'file_path'})

df_train_negative = dataset_train_negative[['path']]
df_train_negative['label'] = 0
df_train_negative = df_train_negative.rename(columns={'path': 'file_path'})

df = pd.concat([df_train_positive, df_train_negative], ignore_index=True)
files = df['file_path'].tolist()
labels = df['label'].tolist()

# Test dataset

df_test_positive = dataset_test_positive[['path']]
df_test_positive['label'] = 1
df_test_positive = df_test_positive.rename(columns={'path': 'file_path'})

df_test_negative = dataset_test_negative[['path']]
df_test_negative['label'] = 0
df_test_negative = df_test_negative.rename(columns={'path': 'file_path'})

df_test = pd.concat([df_test_positive, df_test_negative], ignore_index=True)
files_test = df_test['file_path'].tolist()
labels_test = df_test['label'].tolist()

# Función para cargar audio y procesarlo
def load_audio(file, target_length):
  target_length = int(target_length)

  audio = AudioSegment.from_file(file)

  if config_datos['volume_normalization']:
    current_db = audio.dBFS
    gain = -23 - current_db
    audio = audio.apply_gain(gain)
  
  if config_datos['volume_randomize']:
    audio = audio.apply_gain(random.randint(config_datos['volume_randomize_limits'][0], config_datos['volume_randomize_limits'][1]))

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

print('Converting audio clips!')

# Convertir el dataset a formato Dataset de Hugging Face
dataset = Dataset.from_pandas(pd.DataFrame({'audio': files, 'label': labels}))
dataset_test = Dataset.from_pandas(pd.DataFrame({'audio': files_test, 'label': labels_test}))

# Procesar los audios
processor = Wav2Vec2Processor.from_pretrained(config_datos['processor_path'])

def preprocess_function(examples):
  audio = [load_audio(file, 16000*(config_datos['max_audio_length'] / 1000)) for file in examples['audio']]
  inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
  inputs['labels'] = torch.tensor(examples['label'])
  return inputs

encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["audio"])
encoded_dataset_test = dataset_test.map(preprocess_function, batched=True, remove_columns=["audio"])

# Cargar el modelo y configurar el entrenamiento
if config_datos['base_model_type'] == 'hubert':
  model = HubertForSequenceClassification.from_pretrained(pretrained_model, num_labels=2, ignore_mismatched_sizes=True)
else:
  model = Wav2Vec2ForSequenceClassification.from_pretrained(pretrained_model)

model_train_options = config_datos['model_train_options']

training_args = TrainingArguments(
  output_dir='./model/results',
  evaluation_strategy=model_train_options['evaluation_strategy'],
  save_strategy=model_train_options['save_strategy'],
  per_device_train_batch_size=model_train_options['per_device_train_batch_size'],
  per_device_eval_batch_size=model_train_options['per_device_eval_batch_size'],
  num_train_epochs=model_train_options['num_train_epochs'],
  logging_dir='./model/logs',
  # fp16=True,                    # Opcional
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=encoded_dataset,
  eval_dataset=encoded_dataset_test,  # Puede ser diferente si tienes un conjunto de validación separado
)

train_init_time = time.time()

# Entrenar el modelo
print('Training the model!')
trainer.train()

train_end_time = time.time()

# Guardar el modelo y el procesador después del entrenamiento
print('Saving the model!')
model.save_pretrained("./model/model")
processor.save_pretrained("./model/processor")

file_end_time = time.time()

final_model_config = {
  "your_config": config_datos,
  "file_time": file_end_time - file_init_time,
  "train_time": train_end_time - train_init_time,
  "dataset_conformation": {
    "positive_train_dataset": df_train_positive.shape[0],
    "positive_test_dataset": df_test_positive.shape[0],
    "negative_train_dataset": df_train_negative.shape[0],
    "negative_test_dataset": df_test_negative.shape[0],
    "final_train_dataset": len(files),
    "final_test_dataset": len(files_test)
  }
}

with open("./model/model_config.json", 'w') as file:
  json.dump(final_model_config, file, indent=4)

print('Finished task!')