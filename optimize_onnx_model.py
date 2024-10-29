import onnx
import onnxoptimizer
import argparse

parser = argparse.ArgumentParser(description='Model quantization')
parser.add_argument('-model', type=str, required=False, help='Filename of ONNX model to optimize')
args = parser.parse_args()

base_model_name = 'model.onnx'
if args.model:
  base_model_name = args.model
base_model_path = 'model/' + base_model_name

model_path = base_model_path
model = onnx.load(model_path)

# Aplica optimizaciones al modelo
optimized_model = onnxoptimizer.optimize(model)

# Guarda el modelo optimizado
optimized_model_path = 'model/optimized_' + base_model_name
onnx.save(optimized_model, optimized_model_path)