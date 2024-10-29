import json
import sys
import numpy as np
import os
import pandas as pd

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

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

print('----------------------')
print('Positive train dataset')
print('Total clips:', df_train_positive.shape[0])
print(df_train_positive.head())

print('----------------------')
print('Positive test dataset')
print('Total clips:', df_test_positive.shape[0])
print(df_test_positive.head())

print('----------------------')
print('Negative train dataset')
print('Total clips:', df_train_negative.shape[0])
print(df_train_negative.head())

print('----------------------')
print('Negative test dataset')
print('Total clips:', df_test_negative.shape[0])
print(df_test_negative.head())

print('----------------------')
print('Final train dataset')
print('Total clips:', len(files))
print('Example file:', files[0])

print('----------------------')
print('Final test dataset')
print('Total clips:', len(files_test))
print('Example file:', files_test[0])

print('----------------------')
print('Your config')
print(config_datos)

print('----------------------')