import torch
from transformers import Wav2Vec2ForSequenceClassification, HubertForSequenceClassification
import onnx
import json
import argparse

parser = argparse.ArgumentParser(description='Model conversion to ONNX')
parser.add_argument('-model', type=str, required=False, help='Name of model container folder')
args = parser.parse_args()

base_model_name = 'model'
if args.model:
  base_model_name = args.model
base_model_path = './model/' + base_model_name

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

# Cargar el modelo entrenado
if config_datos['base_model_type'] == 'hubert':
  model = HubertForSequenceClassification.from_pretrained(base_model_path, num_labels=2, ignore_mismatched_sizes=True)
else:
  model = Wav2Vec2ForSequenceClassification.from_pretrained(base_model_path)

# Establecer el modelo en modo evaluación
model.eval()

# Crear un tensor de entrada ficticio con la forma adecuada
dummy_input = torch.randn(1, int(round(16000 * (config_datos['max_audio_length'] / 1000))))  # 1 ejemplo, 3 segundos de audio a 16kHz

# Exportar el modelo a ONNX
onnx_path = './model/'+base_model_name+'.onnx'
torch.onnx.export(
  model, 
  dummy_input,
  onnx_path,
  input_names=['input_values'],
  output_names=['logits'],
  opset_version=14
)

print(f'Model exported to {onnx_path}')