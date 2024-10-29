import onnx
from onnxconverter_common import float16
import argparse

parser = argparse.ArgumentParser(description='Model quantization')
parser.add_argument('-model', type=str, required=False, help='Filename of ONNX model to float16')
args = parser.parse_args()

base_model_name = 'model.onnx'
if args.model:
  base_model_name = args.model
base_model_path = 'model/' + base_model_name

# Ruta al modelo ONNX original
model_path = base_model_path
# Ruta al modelo ONNX convertido a float16
float16_model_path = 'model/float16_' + base_model_name

# Cargar el modelo ONNX original
onnx_model = onnx.load(model_path)

# Convertir el modelo a float16
float16_model = float16.convert_float_to_float16(onnx_model)

# Guardar el modelo convertido a float16
onnx.save(float16_model, float16_model_path)
print(f'Modelo convertido a float16 guardado en {float16_model_path}')