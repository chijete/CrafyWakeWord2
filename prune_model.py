import torch
import torch.nn.utils.prune as prune
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import sys
import os
import shutil

if os.path.isdir('model/pruned_model'):
  shutil.rmtree('model/pruned_model')
if os.path.isdir('model/pruned_processor'):
  shutil.rmtree('model/pruned_processor')

# 1. Cargar el modelo y el procesador desde Hugging Face
model = Wav2Vec2ForSequenceClassification.from_pretrained("model/model")
processor = Wav2Vec2Processor.from_pretrained("model/processor")

prune_amount = 0.5

# 2. Aplicar poda al modelo
for name, module in model.named_modules():
  if isinstance(module, torch.nn.Linear):
    try:
      prune.l1_unstructured(module, name='weight', amount=prune_amount)
      prune.remove(module, 'weight')
      print('Pruned Linear layer')
    except Exception as e:
      print(f"Error pruning Linear layer {name}: {e}")
  elif isinstance(module, torch.nn.Conv1d):
    try:
      prune.l1_unstructured(module, name='weight', amount=prune_amount)
      prune.remove(module, 'weight')
      print('Pruned Conv1d layer')
    except Exception as e:
      print(f"Error pruning Conv1d layer {name}: {e}")

def is_pruned(module):
  return hasattr(module, 'weight_orig')

for name, module in model.named_modules():
  if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) and is_pruned(module):
      print(f"Layer {name} is pruned.")

sys.exit()

# Guardar el modelo y el procesador después del entrenamiento
print('Saving the pruned model!')
model.save_pretrained("./model/pruned_model")
processor.save_pretrained("./model/pruned_processor")

print('Finished!')