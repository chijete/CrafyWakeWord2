from pydub import AudioSegment
import pandas as pd
import sys
import json
import re
import os
from os import listdir
from os.path import isfile, join
import glob
import shutil
import random
import math
import re

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

keyword = config_datos['wake_word']

key_pattern = re.compile("\'(?P<k>[^ ]+)\'")

if not os.path.isdir('datasets/base/positive/') or not os.path.isdir('datasets/base/negative/'):
  print('Must create datasets/base/positive/ and datasets/base/negative/ directories')
  sys.exit()

def list_files(mypath):
  return [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]

def porcentaje_a_db(porcentaje):
  factor_de_volumen = porcentaje / 100.0
  db_reduccion = 20 * math.log10(factor_de_volumen)
  return db_reduccion

noise_test_files = list_files('datasets/noise/noise_test/')
noise_train_files = list_files('datasets/noise/noise_train/')

random.shuffle(noise_train_files)

path_base_positive_dataset = 'datasets/base/positive/'
path_base_negative_dataset = 'datasets/base/negative/'
path_train_positive_dataset = 'datasets/train/positive/'
path_test_positive_dataset = 'datasets/test/positive/'
path_train_negative_dataset = 'datasets/train/negative/'
path_test_negative_dataset = 'datasets/test/negative/'
path_base_positive_generated_dataset = 'datasets/base/positive_generated/'

if os.path.isdir('datasets/train/'):
  shutil.rmtree('datasets/train/')
os.mkdir('datasets/train/')

if os.path.isdir('datasets/test/'):
  shutil.rmtree('datasets/test/')
os.mkdir('datasets/test/')

os.mkdir(path_train_positive_dataset)
os.mkdir(path_test_positive_dataset)
os.mkdir(path_train_negative_dataset)
os.mkdir(path_test_negative_dataset)
os.mkdir(path_train_positive_dataset + 'clips/')
os.mkdir(path_train_negative_dataset + 'clips/')

patron_csv = '*.csv'
archivos_csv = glob.glob(os.path.join(path_base_positive_dataset, patron_csv))
archivos_csv_count = len(archivos_csv)

dataframe_columns = ['path', 'duration']

# Positive dataset

train_positive_df = pd.DataFrame(columns=dataframe_columns)
test_positive_df = pd.DataFrame(columns=dataframe_columns)

positive_clips_count = 0
for csvFilePath in archivos_csv:
  csv_train_data_df = pd.read_csv(csvFilePath)
  positive_clips_count = positive_clips_count + csv_train_data_df.shape[0]

print('Base positive clips:', positive_clips_count)

maxCount_part_of_positive_clips_to_negative = False
if config_datos['part_of_positive_clips_to_negative_proportion'] != False and config_datos['part_of_positive_clips_to_negative_proportion'] > 0:
  maxCount_part_of_positive_clips_to_negative = positive_clips_count * config_datos['part_of_positive_clips_to_negative_proportion']

fpopctn_df = pd.DataFrame(columns=dataframe_columns)
lpopctn_df = pd.DataFrame(columns=dataframe_columns)

positive_rejected_by_keyword = 0
positive_rejected_by_duration = 0

forIndex = 0
forIndex_wov = 0
for csvFilePath in archivos_csv:

  positive_train_data = pd.read_csv(csvFilePath)

  # Mezclar el dataframe
  positive_train_data = positive_train_data.sample(frac=1, random_state=42).reset_index(drop=True)

  for dfIndex, trainElement in positive_train_data.iterrows():

    audio_file_name = os.path.basename(trainElement['path'])
    audio_file_path = path_base_positive_dataset + 'clips/' + audio_file_name

    trainElementTimestamps = json.loads(key_pattern.sub(r'"\g<k>"', trainElement['timestamps']))

    if keyword in trainElementTimestamps:

      # Cargar el archivo de audio
      audio = AudioSegment.from_file(audio_file_path)

      # Calcular los tiempos en milisegundos
      start_time = trainElementTimestamps[keyword]['start'] * 1000
      end_time = trainElementTimestamps[keyword]['end'] * 1000
      total_time = end_time - start_time

      if total_time <= config_datos['max_audio_length'] and total_time >= config_datos['min_audio_length']:

        # Cortar el segmento de audio
        segmento_cortado = audio[start_time:end_time]

        popctn_export_path = path_train_negative_dataset + 'clips/'
        if maxCount_part_of_positive_clips_to_negative != False:
          if forIndex_wov < maxCount_part_of_positive_clips_to_negative:

            if config_datos['first_part_of_positive_clips_to_negative']:

              fpopctn_segment = segmento_cortado
              fpopctn_segment = fpopctn_segment[0:round(total_time/2.5)]

              # Add noise
              w_noise_total_time = random.randint(config_datos['min_audio_length'], config_datos['max_audio_length'])
              noise_volume = porcentaje_a_db(random.randint(config_datos['noise_in_negative_clips_ends'][0], config_datos['noise_in_negative_clips_ends'][1]))
              noise_audio = AudioSegment.from_file(noise_train_files[random.randint(0, len(noise_train_files) - 1)])
              noise_audio_duration = len(noise_audio)
              noise_audio_start_time = round(random.uniform(0, noise_audio_duration - w_noise_total_time - 0.06), 2)
              noise_audio_end_time = noise_audio_start_time + w_noise_total_time
              noise_audio = noise_audio[noise_audio_start_time:noise_audio_end_time]
              noise_audio = noise_audio + noise_volume
              fpopctn_segment = noise_audio.overlay(fpopctn_segment, position=0)

              fpopctn_exportPath = popctn_export_path + 'fpopctn_' + audio_file_name
              fpopctn_segment.export(fpopctn_exportPath, format="wav")
              fpopctn_df = pd.concat([fpopctn_df, pd.DataFrame([{
                'path': fpopctn_exportPath,
                'duration': len(fpopctn_segment)
              }])], ignore_index=True)

            if config_datos['last_part_of_positive_clips_to_negative']:

              lpopctn_segment = segmento_cortado
              lpopctn_segment = lpopctn_segment[round(total_time/3*2):total_time]

              # Add noise
              w_noise_total_time = random.randint(config_datos['min_audio_length'], config_datos['max_audio_length'])
              noise_volume = porcentaje_a_db(random.randint(config_datos['noise_in_negative_clips_ends'][0], config_datos['noise_in_negative_clips_ends'][1]))
              noise_audio = AudioSegment.from_file(noise_train_files[random.randint(0, len(noise_train_files) - 1)])
              noise_audio_duration = len(noise_audio)
              noise_audio_start_time = round(random.uniform(0, noise_audio_duration - w_noise_total_time - 0.06), 2)
              noise_audio_end_time = noise_audio_start_time + w_noise_total_time
              noise_audio = noise_audio[noise_audio_start_time:noise_audio_end_time]
              noise_audio = noise_audio + noise_volume
              position_ms = round(len(noise_audio) - len(lpopctn_segment)) - 1
              lpopctn_segment = noise_audio.overlay(lpopctn_segment, position=position_ms)

              lpopctn_exportPath = popctn_export_path + 'lpopctn_' + audio_file_name
              lpopctn_segment.export(lpopctn_exportPath, format="wav")
              lpopctn_df = pd.concat([lpopctn_df, pd.DataFrame([{
                'path': lpopctn_exportPath,
                'duration': len(lpopctn_segment)
              }])], ignore_index=True)

        for volumePercentVariation in config_datos['positive_volume_variations']:

          for noiseVariationIndex in range(config_datos['positive_noise_variations']):

            variated_segment = segmento_cortado

            variated_segment = variated_segment + porcentaje_a_db(volumePercentVariation)

            # Guardar el segmento cortado en un nuevo archivo
            exportPath = path_train_positive_dataset + 'clips/'
            
            exportPath = exportPath + 'v_' + str(volumePercentVariation) + '_n_' + str(noiseVariationIndex) + '_' + audio_file_name

            # Add noise
            if config_datos['add_noise_in_positive_clips']:
              noise_volume = porcentaje_a_db(random.randint(config_datos['noise_in_positive_clips_ends'][0], config_datos['noise_in_positive_clips_ends'][1]))
              noise_audio = AudioSegment.from_file(noise_train_files[random.randint(0, len(noise_train_files) - 1)])
              noise_audio_duration = len(noise_audio)
              noise_audio_start_time = round(random.uniform(0, noise_audio_duration - total_time - 0.06), 2)
              noise_audio_end_time = noise_audio_start_time + total_time
              noise_audio = noise_audio[noise_audio_start_time:noise_audio_end_time]
              noise_audio = noise_audio + noise_volume
              variated_segment = variated_segment.overlay(noise_audio)

            variated_segment.export(exportPath, format="wav")

            train_positive_df = pd.concat([train_positive_df, pd.DataFrame([{
              'path': exportPath,
              'duration': total_time
            }])], ignore_index=True)

            forIndex = forIndex + 1

        forIndex_wov = forIndex_wov + 1

      else:

        positive_rejected_by_duration = positive_rejected_by_duration + 1
    
    else:

      positive_rejected_by_keyword = positive_rejected_by_keyword + 1

# Prev Negative dataset with positive clips but no words (ptn)

if config_datos['positive_to_negative_out_words_proportion'] > 0:
  max_ptn_clips = forIndex * config_datos['positive_to_negative_out_words_proportion']
  
  ptn_df = pd.DataFrame(columns=dataframe_columns)

  ptnForIndex = 0

  for csvFilePath in archivos_csv:

    ptn_train_data = pd.read_csv(csvFilePath)

    for dfIndex, trainElement in ptn_train_data.iterrows():

      audio_file_name = os.path.basename(trainElement['path'])
      audio_file_path = path_base_positive_dataset + 'clips/' + audio_file_name

      trainElementTimestamps = json.loads(key_pattern.sub(r'"\g<k>"', trainElement['timestamps']))

      for key, value in trainElementTimestamps.items():
        if key != keyword:
          if ptnForIndex < max_ptn_clips:

            # Cargar el archivo de audio
            audio = AudioSegment.from_file(audio_file_path)

            # Calcular los tiempos en milisegundos
            start_time = value['start'] * 1000
            end_time = value['end'] * 1000
            total_time = end_time - start_time

            if total_time <= config_datos['max_audio_length'] and total_time >= config_datos['min_audio_length']:

              # Cortar el segmento de audio
              segmento_cortado = audio[start_time:end_time]

              # Guardar el segmento cortado en un nuevo archivo
              exportPath = path_train_negative_dataset + 'clips/'
              
              exportPath = exportPath + os.path.splitext(audio_file_name)[0] + '_' + re.sub(r'[^a-zA-Z0-9]', '', key) + '_' + str(ptnForIndex) + '.wav'

              # Add noise
              if config_datos['add_noise_in_negative_clips']:
                noise_volume = porcentaje_a_db(random.randint(config_datos['noise_in_positive_clips_ends'][0], config_datos['noise_in_positive_clips_ends'][1]))
                noise_audio = AudioSegment.from_file(noise_train_files[random.randint(0, len(noise_train_files) - 1)])
                noise_audio_duration = len(noise_audio)
                noise_audio_start_time = round(random.uniform(0, noise_audio_duration - total_time - 0.06), 2)
                noise_audio_end_time = noise_audio_start_time + total_time
                noise_audio = noise_audio[noise_audio_start_time:noise_audio_end_time]
                noise_audio = noise_audio + noise_volume
                segmento_cortado = segmento_cortado.overlay(noise_audio)

              segmento_cortado.export(exportPath, format="wav")

              ptn_df = pd.concat([ptn_df, pd.DataFrame([{
                'path': exportPath,
                'duration': total_time
              }])], ignore_index=True)

              ptnForIndex = ptnForIndex + 1

# Generated audio clips dataset -> positive dataset

generatedsForIndex = 0

if os.path.isdir(path_base_positive_generated_dataset):

  archivos_csv = glob.glob(os.path.join(path_base_positive_generated_dataset, patron_csv))
  archivos_csv_count = len(archivos_csv)

  print('Generated positive datasets csv files:', archivos_csv_count)

  max_generated_clips = False
  if config_datos['positive_generated_max_proportion_base_positive'] > 0:
    max_generated_clips = forIndex * config_datos['positive_generated_max_proportion_base_positive']

  min_generated_audio_duration = False
  if config_datos['positive_generated_min_clip_duration'] > 0:
    min_generated_audio_duration = config_datos['positive_generated_min_clip_duration']

  if archivos_csv_count > 0:

    for csvFilePath in archivos_csv:

      generated_dataset = pd.read_csv(csvFilePath)

      for dfIndex, trainElement in generated_dataset.iterrows():

        if max_generated_clips == False or generatedsForIndex < max_generated_clips:

          audio_file_name = os.path.basename(trainElement['path'])
          audio_file_path = path_base_positive_generated_dataset + 'clips/' + audio_file_name

          if keyword in trainElement['sentence']:

            # Cargar el archivo de audio
            audio = AudioSegment.from_file(audio_file_path)

            # Calcular los tiempos en milisegundos
            start_time = 0
            end_time = len(audio)
            total_time = end_time - start_time

            if total_time <= config_datos['max_audio_length'] and total_time >= config_datos['min_audio_length']:

              if min_generated_audio_duration == False or total_time >= min_generated_audio_duration:

                segmento_cortado = audio

                exportPath = path_train_positive_dataset + 'clips/' + audio_file_name

                # Add noise
                if config_datos['add_noise_in_positive_clips']:
                  noise_volume = porcentaje_a_db(random.randint(config_datos['noise_in_positive_clips_ends'][0], config_datos['noise_in_positive_clips_ends'][1]))
                  noise_audio = AudioSegment.from_file(noise_train_files[random.randint(0, len(noise_train_files) - 1)])
                  noise_audio_duration = len(noise_audio)
                  noise_audio_start_time = round(random.uniform(0, noise_audio_duration - total_time - 0.06), 2)
                  noise_audio_end_time = noise_audio_start_time + total_time
                  noise_audio = noise_audio[noise_audio_start_time:noise_audio_end_time]
                  noise_audio = noise_audio + noise_volume
                  segmento_cortado = segmento_cortado.overlay(noise_audio)

                segmento_cortado.export(exportPath, format="wav")

                train_positive_df = pd.concat([train_positive_df, pd.DataFrame([{
                  'path': exportPath,
                  'duration': total_time
                }])], ignore_index=True)

                forIndex = forIndex + 1
                generatedsForIndex = generatedsForIndex + 1

    print('Saved positive generated clips:', generatedsForIndex)

# Mezclar los ejemplos positivos
train_positive_df = train_positive_df.sample(frac=1, random_state=42).reset_index(drop=True)

test_files_count = round(forIndex * config_datos['test_percentage'])
if test_files_count > 0:
  test_positive_df = train_positive_df.tail(test_files_count)
  train_positive_df = train_positive_df.drop(train_positive_df.tail(test_files_count).index)

print('train_positive_df:')
print(train_positive_df.head())

print('test_positive_df:')
print(test_positive_df.head())

train_positive_df.to_csv(path_train_positive_dataset + 'dataset.csv', index=False)
test_positive_df.to_csv(path_test_positive_dataset + 'dataset.csv', index=False)

print('Saved positive clips:', forIndex)
print('Rejected positive clips by duration:', positive_rejected_by_duration)
print('Rejected positive clips by keyword:', positive_rejected_by_keyword)

print('Saved positive clips of not-generated:', (forIndex - generatedsForIndex))

# Negative dataset

max_negative_clips = False
if config_datos['positive_negative_fixed_proportion'] > 0:
  max_negative_clips = forIndex * config_datos['positive_negative_fixed_proportion']

archivos_csv = glob.glob(os.path.join(path_base_negative_dataset, patron_csv))
archivos_csv_count = len(archivos_csv)

train_negative_df = pd.DataFrame(columns=dataframe_columns)
test_negative_df = pd.DataFrame(columns=dataframe_columns)

negative_clips_count = 0
for csvFilePath in archivos_csv:
  csv_train_data_df = pd.read_csv(csvFilePath)
  negative_clips_count = negative_clips_count + csv_train_data_df.shape[0]

print('Base negative clips:', negative_clips_count)
print('Max negative clips:', max_negative_clips)

negative_rejected_by_keyword = 0

forIndex = 0
for csvFilePath in archivos_csv:

  negative_train_data = pd.read_csv(csvFilePath)

  for dfIndex, trainElement in negative_train_data.iterrows():

    if max_negative_clips == False or forIndex <= max_negative_clips:

      if keyword not in trainElement['sentence']:

        audio_file_name = os.path.basename(trainElement['path'])
        audio_file_path = path_base_negative_dataset + 'clips/' + audio_file_name

        # Cargar el archivo de audio
        audio = AudioSegment.from_file(audio_file_path)

        # Calcular los tiempos en milisegundos
        total_time = trainElement['duration']
        segmento_cortado = audio

        for rIndex in range(config_datos['negative_random_variations']):

          variated_segment = segmento_cortado

          if trainElement['duration'] > config_datos['max_audio_length']:

            new_duration = round(random.uniform(config_datos['min_audio_length'], config_datos['max_audio_length'] - 0.06), 2)

            start_time = round(random.uniform(0, len(variated_segment) - new_duration - 0.06), 2)
            end_time = start_time + new_duration
            total_time = end_time - start_time

            variated_segment = variated_segment[start_time:end_time]

          else:

            new_duration = round(random.uniform(config_datos['min_audio_length'], len(variated_segment) - 0.06), 2)

            start_time = round(random.uniform(0, len(variated_segment) - new_duration - 0.06), 2)
            end_time = start_time + new_duration
            total_time = end_time - start_time

            variated_segment = variated_segment[start_time:end_time]
          
          if total_time <= config_datos['max_audio_length'] and total_time >= config_datos['min_audio_length']:
          
            # Guardar el segmento cortado en un nuevo archivo
            exportPath = path_train_negative_dataset + 'clips/'
            
            exportPath = exportPath + 'r' + str(rIndex) + '_' + audio_file_name

            # Add noise
            if config_datos['add_noise_in_negative_clips']:
              noise_volume = porcentaje_a_db(random.randint(config_datos['noise_in_negative_clips_ends'][0], config_datos['noise_in_negative_clips_ends'][1]))
              noise_audio = AudioSegment.from_file(noise_train_files[random.randint(0, len(noise_train_files) - 1)])
              noise_audio_duration = len(noise_audio)
              noise_audio_start_time = round(random.uniform(0, noise_audio_duration - total_time - 0.06), 2)
              noise_audio_end_time = noise_audio_start_time + total_time
              noise_audio = noise_audio[noise_audio_start_time:noise_audio_end_time]
              noise_audio = noise_audio + noise_volume
              variated_segment = variated_segment.overlay(noise_audio)

            variated_segment.export(exportPath, format="wav")

            train_negative_df = pd.concat([train_negative_df, pd.DataFrame([{
              'path': exportPath,
              'duration': total_time
            }])], ignore_index=True)

            forIndex = forIndex + 1
      
      else:

        negative_rejected_by_keyword = negative_rejected_by_keyword + 1

if config_datos['vanilla_noise_in_negative_dataset_proportion'] > 0:
  vn_max_clips = forIndex * config_datos['vanilla_noise_in_negative_dataset_proportion']

  print('Max vanilla noise clips:', vn_max_clips)

  noiseForIndex = 0
  for noise_clip in noise_train_files:
    if noiseForIndex < vn_max_clips:

      audio_file_name = os.path.basename(noise_clip)
      audio_file_path = noise_clip

      # Cargar el archivo de audio
      audio = AudioSegment.from_file(audio_file_path)

      # Calcular los tiempos en milisegundos
      total_time = len(audio)
      segmento_cortado = audio

      if total_time > config_datos['max_audio_length']:

        new_duration = round(random.uniform(config_datos['max_audio_length'] / 2, config_datos['max_audio_length'] - 0.06), 2)

        start_time = round(random.uniform(0, config_datos['max_audio_length'] - new_duration - 0.06), 2)
        end_time = start_time + new_duration
        total_time = end_time - start_time

        segmento_cortado = audio[start_time:end_time]

      if total_time <= config_datos['max_audio_length'] and total_time >= config_datos['min_audio_length']:
      
        # Guardar el segmento cortado en un nuevo archivo
        exportPath = path_train_negative_dataset + 'clips/'
        
        exportPath = exportPath + audio_file_name

        segmento_cortado.export(exportPath, format="wav")

        train_negative_df = pd.concat([train_negative_df, pd.DataFrame([{
          'path': exportPath,
          'duration': total_time
        }])], ignore_index=True)

        noiseForIndex = noiseForIndex + 1

  print('Vanilla noise clips added to negative dataset:', noiseForIndex)

if config_datos['positive_to_negative_out_words_proportion'] > 0:
  print('PTN clips:', ptn_df.shape[0])

  train_negative_df = pd.concat([train_negative_df, ptn_df], ignore_index=True)

  forIndex = forIndex + ptnForIndex

if fpopctn_df.shape[0] > 0:
  print('fpopctn_df clips:', fpopctn_df.shape[0])
  train_negative_df = pd.concat([train_negative_df, fpopctn_df], ignore_index=True)
  forIndex = forIndex + fpopctn_df.shape[0]

if lpopctn_df.shape[0] > 0:
  print('lpopctn_df clips:', lpopctn_df.shape[0])
  train_negative_df = pd.concat([train_negative_df, lpopctn_df], ignore_index=True)
  forIndex = forIndex + fpopctn_df.shape[0]

# Mezclar los ejemplos negativos
train_negative_df = train_negative_df.sample(frac=1, random_state=42).reset_index(drop=True)

test_files_count = round(forIndex * config_datos['test_percentage'])
if test_files_count > 0:
  test_negative_df = train_negative_df.tail(test_files_count)
  train_negative_df = train_negative_df.drop(train_negative_df.tail(test_files_count).index)

print('train_negative_df:')
print(train_negative_df.head())

print('test_negative_df:')
print(test_negative_df.head())

train_negative_df.to_csv(path_train_negative_dataset + 'dataset.csv', index=False)
test_negative_df.to_csv(path_test_negative_dataset + 'dataset.csv', index=False)

print('Saved negative clips:', forIndex)
print('Rejected negative clips by keyword:', negative_rejected_by_keyword)