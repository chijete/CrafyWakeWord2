from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse

parser = argparse.ArgumentParser(description='Model quantization')
parser.add_argument('-level', type=int, default=0, required=False, help='Quantization level (0 = less aggressive, 1 = more aggressive)')
parser.add_argument('-model', type=str, required=False, help='Filename of ONNX model to quantize')
args = parser.parse_args()

quant_type_str = '0'
if args.level == 0:
  quant_type = QuantType.QUInt8
else:
  quant_type = QuantType.QInt8
  quant_type_str = '1'
# Options:
# QuantType.QUInt8 (Cuantización a enteros sin signo de 8 bits. Menos agresiva y puede ser más adecuada para algunos modelos con activaciones no negativas.)
# QuantType.QInt8 (Cuantización a enteros con signo de 8 bits. Más agresiva y generalmente más eficiente en términos de reducción de tamaño.)

base_model_name = 'model.onnx'
if args.model:
  base_model_name = args.model
base_model_path = 'model/' + base_model_name

# Ruta al modelo ONNX optimizado
optimized_model_path = base_model_path
# Ruta al modelo ONNX cuantizado
quantized_model_path = 'model/quant_'+quant_type_str+'_'+base_model_name

# Cuantizar el modelo optimizado
quantize_dynamic(optimized_model_path, quantized_model_path, weight_type=quant_type)

print(f'Modelo cuantizado guardado en {quantized_model_path}')