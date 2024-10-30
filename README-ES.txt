Generación del dataset base:
Se hace con CrafyWakeWord (la versión 1: https://github.com/chijete/CrafyWakeWord)
Seguir estos pasos:
1. Se tienen que seguir todos los pasos del README https://github.com/chijete/CrafyWakeWord?tab=readme-ov-file#create-your-own-model
hasta el 5, eso para preparar el entorno.
2. En your_config.json, "wake_words" debe contener la palabra con la que vamos a entrenar el modelo.
2.1. Se debería añadir un CUSTOM DATASET si no hay suficientes ejemplos de la palabra wakeword en el corpus
de Mozilla. Instrucciones: https://github.com/chijete/CrafyWakeWord?tab=readme-ov-file#custom-datasets
3. python dataset_generation.py
4. python v2_generate_negative_dataset.py -limit (Límite de clips a incluir, default 5000)
5. python align.py
6. python align_manage.py
7. python v2_generate_positive_dataset.py
8. python generate_tts_clips.py -word {wake_word}
9. Copiar directorios desde la versión 1 a la 2:
CrafyWakeWord/v2_dataset/negative/ > CrafyWakeWord2/datasets/base/negative/
CrafyWakeWord/v2_dataset/positive/ > CrafyWakeWord2/datasets/base/positive/
CrafyWakeWord/dataset/generated/{wake_word}/ > CrafyWakeWord2/datasets/base/positive_generated/clips/
CrafyWakeWord/dataset/generated/{wake_word}.csv > CrafyWakeWord2/datasets/base/positive_generated/dataset.csv

Ahí quedan en CrafyWakeWord2:
datasets/
  base/ (Creados en los pasos anteriores)
    negative/
    positive/
    positive_generated/
  noise/ (Queda en el repositorio)

Usar CrafyWakeWord2:
1. Configurar your_config.json
2. python convert_datasets.py (crea datasets/test/ y datasets/train/)
3. python check_dataset.py (opcional, muestra información de los datasets)
4. python train.py (entrena el modelo: hace un finetuning del your_config.json > "base_model")
5. python test_model.py (opcional, prueba el modelo entrenado)
6. python convert_model_to_onnx.py (convierte el modelo de HuggingFace a ONNX)
7. python test_model_onnx.py -model {Filename del archivo .onnx a evaluar, que está dentro de model/} (opcional, prueba el modelo entrenado en formato ONNX)
8. python optimize_onnx_model.py -model {Filename del archivo .onnx a optimizar, que está dentro de model/} (opcional, optimiza el modelo ONNX)
9. python quant_onnx_model.py -model {Filename del archivo .onnx a cuantizar, que está dentro de model/} (opcional, aplica cuantización al modelo ONNX)

Con distilhubert descubrí que lo más eficiente es unicamente optimizar el modelo,
la cuantización lo vuelve más lento a cambio de ahorrar peso (MB), aunque eso no es muy
importante si se descarga el modelo al dispositivo del usuario.

Al final, el modelo entrenado en sus diferentes variantes estará en model/