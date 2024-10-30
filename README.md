# CrafyWakeWord2

CrafyWakeWord2 it's a library focused on AI-based wake word recognition.

This is the second version of CrafyWakeWord. You can find the first version here: [https://github.com/chijete/CrafyWakeWord](https://github.com/chijete/CrafyWakeWord)

This second version requires using some files from the first version. More information below.

## ⭐ Features and functions

*   Custom wake word recognition.
*   Multiple language support.
*   Models portable to other platforms. ONNX universal format.
*   Javascript execution supported.
*   Step by step explanation.

## How it works

This library finetunes a Transformers-based audio vectorization model, training it to classify audio into two categories:

*   The wake word was said.
*   The wake word was not said.

This classification model can then be used in real time to detect when the user says the wake word through the microphone.

It can also be used to classify pre-recorded audio.

## Demo

You can see an online demo here: [https://crafywakeword2.netlify.app/](https://crafywakeword2.netlify.app/)

Wait until you are prompted for microphone access, and then say “almeja”.

# Create your own model

With this tool you can create your custom wake word detection model. For example, you can create a model to detect when the user says the word "banana", and then run your own code accordingly.

## Prerequisites

*   Have [Python](https://www.python.org/downloads/) 3 installed.
*   Have [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download#) installed (**optional**).
*   Have a verified Google Cloud account (we will use the [Google Cloud Text-to-Speech API](https://console.cloud.google.com/apis/library/texttospeech.googleapis.com) to improve the dataset, more information below; the free plan is enough).

## 1\. Download voice dataset

The first step is to obtain a dataset of transcribed audios. In this library we will use Mozilla Common Voice to obtain the dataset.

Follow these steps:

1.  Access to [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)
2.  Select the target language from the Language selector.
3.  Select the last "Common Voice Corpus" version (Do not select "Delta Segment").
4.  Enter an email, accept the terms and download the file.

## 2\. Clone repositories

1.  V1 repository:
    1.  Clone [v1 repository](https://github.com/chijete/CrafyWakeWord) to a folder on your computer using [git](https://git-scm.com/), or download and unzip this repository using Github's "Code > Download ZIP" option.
    2.  You should get a folder called `CrafyWakeWord`.
    3.  Unzip the downloaded Mozilla Common Voice file and copy the "cv-corpus-..." folder to `CrafyWakeWord/corpus/`.
2.  V2 repository:
    1.  Clone this repository to a folder on your computer using [git](https://git-scm.com/), or download and unzip this repository using Github's "Code > Download ZIP" option.
    2.  You should get a folder called `CrafyWakeWord2`.

## 3\. Install dependencies

Run this commands in your terminal (**optional**: conda activate first or Anaconda terminal):

*   `pip install librosa textgrid torchsummary ffmpeg-python pocketsphinx fastprogress chardet PyAudio clang pgvector hdbscan initdb speechbrain`
*   `pip install --upgrade google-cloud-texttospeech`
*   `pip install --only-binary :all: pynini` or `conda install conda-forge::pynini`
*   `conda install -c conda-forge kalpy`
*   `pip install montreal-forced-aligner`
*   `conda install -c conda-forge sox`
*   `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` (Installing PyTorch with CPU, complete instructions on [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) - you can use GPU acceleration).
*   `pip install transformers datasets pydub onnx onnxruntime onnxmltools onnxconverter_common onnxoptimizer`
*   `pip install transformers[torch]`

**Install FFmpeg** from [https://www.ffmpeg.org/download.html](https://www.ffmpeg.org/download.html)

**In Windows:**

1.  Open [https://www.gyan.dev/ffmpeg/builds/#release-builds](https://www.gyan.dev/ffmpeg/builds/#release-builds)
2.  Download `ffmpeg-release-essentials.zip`
3.  Unzip downloaded file in `C:\ffmpeg`
4.  Open the Windows Control Panel.
5.  Click on "System and Security".
6.  Select "System".
7.  In the "Advanced system settings" window, click on the "Environment Variables" button under the "Advanced" tab.
8.  In the "System variables" section, look for the "Path" variable and click "Edit...".
9.  Add the path to the FFmpeg bin directory to the end of the list. In this case: `C:\ffmpeg\bin`
10.  Save changes.

**Install PostgreSQL** from [https://www.enterprisedb.com/downloads/postgres-postgresql-downloads](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) When the installation is finished, add PostgreSQL to the System Path:

**In Windows:**

1.  Open the Windows Control Panel.
2.  Click on "System and Security".
3.  Select "System".
4.  In the "Advanced system settings" window, click on the "Environment Variables" button under the "Advanced" tab.
5.  In the "System variables" section, look for the "Path" variable and click "Edit...".
6.  Add the path to the PostgreSQL directory to the end of the list. For example, the path might be something like `"C:\Program Files\PostgreSQL\version\bin"` (replace "version" with the version of PostgreSQL you installed).
7.  Save changes.

**When finished, close the terminal and reopen it to apply the changes.**

## 4\. Download aligner model

We will use [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) to align the audio files from the Mozilla Common Voice dataset. Follow these steps:

1.  Search for an Acoustic model for your model's target language here: [https://mfa-models.readthedocs.io/en/latest/acoustic/index.html](https://mfa-models.readthedocs.io/en/latest/acoustic/index.html)
2.  On the Acoustic model details page, in the Installation section, click "download from the release page".
3.  At the bottom of the page on Github, in the Assets section, click on the zip file (the first one in the list) to download it.
4.  Return to the Acoustic model page, and in the Pronunciation dictionaries section, click on the first one in the list.
5.  On the Pronunciation dictionary details page, in the Installation section, click "download from the release page".
6.  At the bottom of the page on Github, in the Assets section, click on the dict file (the first one in the list) to download it.
7.  Copy the two downloaded files to the `CrafyWakeWord/mfa/` folder within the directory where you cloned the repository v1.

## 5\. Edit config file of v1

Edit `CrafyWakeWord/your_config.json` file:

*   `"common_voice_datapath"` is the path, relative to the root directory, where the downloaded Mozilla Common Voice files are located. Example: `"common_voice_datapath": "corpus/cv-corpus-15.0-2023-09-08/en/"`
*   `"wake_words"` is the list of words that your model will learn to recognize. For this version 2, it should be an array with a single string, which has the wake word in lower case. Example: `["banana"]`
*   `"google_credentials_file"` is the path, relative to the root directory, where your Google Cloud acccess credentials file is located. You can learn how to get your account credentials JSON file in this help article: [https://cloud.google.com/iam/docs/keys-create-delete#creating](https://cloud.google.com/iam/docs/keys-create-delete#creating) . You can paste the credentials file in the root directory where you cloned the repository. Example: `"cloud_credentials.json"`
*   `"mfa_DICTIONARY_PATH"` is the path, relative to the root directory, where your downloaded Montreal Forced Aligner Pronunciation dictionary file is located.
*   `"mfa_ACOUSTIC_MODEL_PATH"` is the path, relative to the root directory, where your downloaded Montreal Forced Aligner Acoustic model file is located.
*   `"dataset_language"` is the [ISO 639-1 code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of the target language. Example: `"en"`
*   `"window_size_ms"` _for this version 2, this index is irrelevant._
*   `"train_epochs"` _for this version 2, this index is irrelevant._
*   `"add_vanilla_noise_to_negative_dataset"` _for this version 2, this index is irrelevant._
*   `"voices_generation_with_google"` _for this version 2, this index is irrelevant._
*   `"custom_dataset_path"` (string or empty string) the path to the directory of your custom dataset. You can get more information in the "Custom datasets" section.
*   `"tts_generated_clips"` config of clips generation with the Google Cloud Text-to-Speech API. Leave it as default or read the "Improving model effectiveness" section.
    *   `"rate"` Speed ​​range of the voices of the generated audios (start, stop and step for np.arange). start and stop: min 0.25, max 4.0.
    *   `"pitch"` Pitch ​​range of the voices of the generated audios (start, stop and step for np.arange). start and stop: min -20.0, max 20.0.

## 6\. Prepare base dataset

We will use the [CrafyWakeWord v1](https://github.com/chijete/CrafyWakeWord) repository to generate the base dataset.

1.  Open your terminal.
2.  Navigate to `CrafyWakeWord/`
3.  Run `python dataset_generation.py`
4.  Run `python v2_generate_negative_dataset.py`
    1.  You can add the `limit` attribute to set the limit of clips that will be included in the negative dataset.
    2.  Example: `python v2_generate_negative_dataset.py -limit 6000`
    3.  Default limit is 5000.
5.  Run `python align.py`
6.  Run `python align_manage.py`
7.  Run `python v2_generate_positive_dataset.py`
8.  Run `python generate_tts_clips.py -word **{wake_word}**` Replace `{wake_word}` with the wake word you are training the model for. Example: `python generate_tts_clips.py -word banana`
9.  Copy files and directories from `CrafyWakeWord/` to `CrafyWakeWord2/`
    1.  `CrafyWakeWord/v2_dataset/negative/` -> `CrafyWakeWord2/datasets/base/negative/`
    2.  `CrafyWakeWord/v2_dataset/positive/` -> `CrafyWakeWord2/datasets/base/positive/`
    3.  `CrafyWakeWord/dataset/generated/**{wake_word}**/` -> `CrafyWakeWord2/datasets/base/positive_generated/clips/`
    4.  `CrafyWakeWord/dataset/generated/**{wake_word}**.csv` -> `CrafyWakeWord2/datasets/base/positive_generated/dataset.csv` (You must rename the file from "`**{wake_word}**.csv`" to "`dataset.csv`").

## 7\. Edit config file of v2

Edit `CrafyWakeWord2/your_config.json` file:

*   `"wake_word"` (string) The wake word that the model will be trained for.
*   `"max_audio_length"` (int) Maximum duration in milliseconds of the model's input audio clip.
*   `"min_audio_length"` (int) Minimum duration in milliseconds of the model's input audio clip.
*   `"test_percentage"` (float) Percentage of the base dataset that will be used to test the model. The remainder will be used for training. Example: 0.1 means that 10% of the base dataset will be used for testing.
*   `"positive_negative_fixed_proportion"` (float) Maximum proportion of the negative dataset with respect to the positive one. Example: 1.8 means that the negative dataset will have a maximum number of clips up to 180% of the clips that the positive dataset has. If set to 0, the number of clips in the negative dataset will not be limited.
*   `"positive_generated_max_proportion_base_positive"` (float) Maximum proportion of the positive dataset generated with TTS with respect to the positive dataset. Exanple: 0.9 means that the positive dataset generated with TTS will have a maximum number of clips up to 90% of the clips that the positive dataset has. If set to 0, the number of clips in the positive dataset generated with TTS will not be limited.
*   `"positive_generated_min_clip_duration"` (int) Minimum audio clip duration in milliseconds for positive dataset generated with TTS clips.
*   `"positive_to_negative_out_words_proportion"` (float) Maximum proportion of the PTN dataset with respect to the positive dataset (without adding the positive dataset generated with TTS). The PTN dataset consists of audio clips that contain a word that is not the wake word, and will be used to add to the negative dataset. The source of these clips is the original positive dataset. Example: 0.8 means that the PTN dataset will have a maximum number of clips up to 80% of the clips that the positive dataset has. If set to 0, the PTN dataset will not be generated.
*   `"positive_volume_variations"` (array of int) Volume variations of the original positive dataset. Volume percentages of the original clip. The number of elements in this array is a multiplier of the number of clips in the final positive dataset. Example: \[50, 100, 150\] means that the final positive dataset will include all clips from the original positive dataset in three volume variants: 50%, 100% and 150%.
*   `"positive_noise_variations"` (int) Number of noise variations in the original positive dataset. Minimum: 1 (a single noise is applied to each clip). A larger number will create more variations of each clip with different background noises. If the background noise function is disabled, this value must still be set to 1.
*   `"negative_random_variations"` (int) Number of random segments from each clip in the original negative dataset to add to the final negative dataset. These random segments contain incomplete sentences, noises or silences, and are useful for reducing false positives. If set to 0, these random segments will not be added to the final negative dataset.
*   `"first_part_of_positive_clips_to_negative"` (int) Percentage of duration of the initial segment of each clip of the positive dataset that will be added to the final negative dataset. Example: 35 means that the initial 35% duration segment of each audio from the positive dataset will be added as a clip from the negative dataset. These pieces from the beginning of clips from the positive dataset sent to the negative dataset help reduce false positives with words that start the same way. If set to 0, these clip pieces will not be added to the negative dataset.
*   `"last_part_of_positive_clips_to_negative"` (int) Percentage of duration of the final segment of each clip of the positive dataset that will be added to the final negative dataset. Example: 35 means that the final 35% duration segment of each audio from the positive dataset will be added as a clip from the negative dataset. These pieces from the end of clips from the positive dataset sent to the negative dataset help reduce false positives with words that end the same way. If set to 0, these clip pieces will not be added to the negative dataset.
*   `"part_of_positive_clips_to_negative_proportion"` (float) Maximum proportion of the number of clips from the positive dataset whose starts and ends (configured in the `"first_part_of_positive_clips_to_negative"` and `"last_part_of_positive_clips_to_negative"` indices) will be added to the negative dataset with respect to the original positive dataset. Setting this to 0 will not limit the number of start and end segments that will be added to the negative dataset.
*   `"volume_normalization"` (bool) Always leave false.
*   `"volume_randomize"` (bool) If set to true, each input clip for training (both the positive and negative dataset) will have a random volume variation applied.
*   `"volume_randomize_limits"` (array of int) Minimum and maximum limit of the gain variation that will be randomly applied to each input clip. `AudioSegment.apply_gain(random.randint(**{minimum}**, **{maximum}**))`. Only required if `"volume_randomize"` is true.
*   `"add_noise_in_positive_clips"` (bool) If set to true, background noise will be added to clips in the positive dataset.
*   `"noise_in_positive_clips_ends"` (array of int) Minimum and maximum percentage of background noise volume for the positive dataset.
*   `"add_noise_in_negative_clips"` (bool) If set to true, background noise will be added to clips in the negative dataset.
*   `"noise_in_negative_clips_ends"` (array of int) Minimum and maximum percentage of background noise volume for the negative dataset.
*   `"vanilla_noise_in_negative_dataset_proportion"` (float) Maximum proportion of vanilla noise clips that will be added to the final negative dataset with respect to the original negative dataset. Example: 0.05 means that a maximum of vanilla noise clips of up to 5% of the total clips that the original negative dataset has will be added to the negative dataset. If set to 0, vanilla noise will not be added to the negative dataset.
*   `"base_model"` (string) Path of the base wav2vec2 or hubert model on which finetuning will be done. You can find models on [HuggingFace](https://huggingface.co/models?sort=trending&search=wav2vec2).
*   `"processor_path"` (string) Path of the processor for wav2vec2 or hubert.
    *   Example for hubert: "facebook/hubert-large-ls960-ft".
    *   Example for wav2vec2: "facebook/wav2vec2-large-960h".
*   `"base_model_type"` (string) Model type. Allowed values: `"hubert"`, `"wav2vec2"`. Must match `"base_model"` and `"processor_path"`.
*   `"model_train_options"` Training settings.
    *   `"evaluation_strategy"` Evaluation strategy.
    *   `"save_strategy"` Save strategy.
    *   `"per_device_train_batch_size"` Number of training modules. If GPU is used, it is equivalent to the number of GB of VRAM that the model will use to train (approximately). If the VRAM of the GPU is 4GB, set it to 4. A larger number means a greater load on the device, but a faster training.
    *   `"per_device_eval_batch_size"` Number of eval modules. Set the same value as in `"per_device_train_batch_size"`.
    *   `"num_train_epochs"` Number of training epochs. A higher number fits the model more and a lower number underfits it. More epochs is more time the training takes.

## 8\. Prepare final dataset

1.  Open your terminal.
2.  Navigate to `CrafyWakeWord2/`
3.  Run `python convert_datasets.py`
4.  **Optional**: run `python check_dataset.py` (shows final dataset information).

## 9\. Train the model

1.  Open your terminal.
2.  Navigate to `CrafyWakeWord2/`
3.  Run `python train.py`
4.  **Optional**: run `python test_model.py` (test trained model).

The trained model will be saved in: `CrafyWakeWord2/model/model/`

## 10\. Convert model to ONNX

1.  Open your terminal.
2.  Navigate to `CrafyWakeWord2/`
3.  Run `python convert_model_to_onnx.py`
4.  **Optional**: run `python test_model_onnx.py` (test trained ONNX model).

The converted model will be saved in: `CrafyWakeWord2/model/model.onnx`

## 11\. Model optimization

You can optimize and apply quantization to the model to make it more efficient at the time of inference.

### Optimization

It applies optimizations to the ONNX model to improve its inference speed, but does not significantly reduce its size (MB).

Run `python optimize_onnx_model.py -model **{onnx_model_filename}**`

You must replace **{onnx\_model\_filename}** with the name of the ONNX format model file stored within `CrafyWakeWord2/model/`. Example: `python optimize_onnx_model.py -model model.onnx`

The optimized model will be saved in: `CrafyWakeWord2/model/optimized_**{onnx_model_filename}**`

### Quantization

[What is Quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)

It applies quantization to the model, achieving a significant reduction in its size (MB), but losing precision.

Run `python quant_onnx_model.py -model **{onnx_model_filename}**`

You must replace **{onnx\_model\_filename}** with the name of the ONNX format model file stored within `CrafyWakeWord2/model/`. Example: `python quant_onnx_model.py -model model.onnx`

The quanted model will be saved in: `CrafyWakeWord2/model/quant_**{onnx_model_filename}**`

# Custom datasets

To improve model training you can add a custom dataset.

The dataset must have a format similar to Mozilla Common Voice: an audio dataset with its corresponding transcription.

To add a custom dataset you must create a directory in the root of the project with the following structure:

*   `clips/` **(mandatory)** a directory containing all the audio clips in the dataset in MP3 format.
*   `train.csv` **(mandatory)** a table in CSV format with the columns "path" and "sentence". In the "path" column a string must be entered with the full name of the audio clip file (example: "audio\_123.mp3"), audio clips must be saved inside the `CrafyWakeWord/clips/` folder; and in the "sentence" column a string must be entered with the complete transcription of the audio clip (example: "Ducks can fly"). The audio clips that will be used for training must be listed in this file.
*   `dev.csv` **(optional)** same structure as `train.csv`. The audio clips that will be used for dev must be listed in this file.
*   `test.csv` **(optional)** same structure as `train.csv`. The audio clips that will be used for test must be listed in this file.

To use the custom dataset, before performing "6. Prepare base dataset", the value of `"custom_dataset_path"` in `CrafyWakeWord/your_config.json` must be set to the path of the directory where the custom dataset is located (relative to the root directory). Example: "custom\_dataset/". If you want not to use a custom dataset, then set the value of `"custom_dataset_path"` to an empty string.

The clips whose transcripts contain the wake word will be added to the positive dataset, while those that do not contain it will be added to the negative dataset.

# Improving model effectiveness

There are several ways to improve the effectiveness of the model.

They all involve adjusting the settings before training.

## Base dataset quality, diversity and richness

As in any deep learning model, it is very important that the examples we use to train it (dataset) are as diverse and of high quality as possible.

It is necessary that the positive dataset has a large number of clips in which the wake word is said, including various voices, tones, recording methods, etc.

A quick way to increase the number of clips in the positive dataset is using Text-to-Speech. This library has the possibility of generating audio clips in which the wake word is said using [Google Cloud Text-to-Speech API](https://console.cloud.google.com/apis/library/texttospeech.googleapis.com).

In `CrafyWakeWord/your_config.json` > `"tts_generated_clips"` you can configure the speed and pitch variations that the artificially generated voices will have. This will be useful to obtain more varied examples. And many clips quickly.

And just as it is important to have many examples of what the model should recognize, it is also important to teach it what not to recognize.

That is why the negative dataset must have examples of sounds, words, noises, phrases, segments of words that are not the wake word.

## Base model and processor

Using a better base model on which finetuning will be done and an appropriate processor will also improve the effectiveness and speed of the final model.

## Try different configurations

Finding better recipes for the final model is often a matter of trial and error.

In `CrafyWakeWord2/your_config.json` you can modify settings and retrain the model to achieve better results.

# Continuous wake word detection

With CrafyWakeWord2 you can train a model capable of categorizing an audio input into two categories: the wake word was said (1), and the wake word was not said (0).

But you probably want to constantly listen to the user and trigger some action when they say the wake word.

To achieve this, you must follow this scheme:

1.  Continuously listen to the user's microphone.
2.  Detect when the user starts to speak (Voice Activity Detection, or VAD).
3.  Detect when the user finishes speaking.
4.  Get the audio clip of the user speaking.
5.  Cut the first segment of the audio, with a maximum duration of `CrafyWakeWord2/your_config.json` > `"max_audio_length"`.
6.  Use the audio clip to make inference to the trained model. (To make the inference you will have to convert the audio and sample it at 16000 Hz. The most optimal thing is to do it using FFmpeg, since other methods or libraries could give different results than what the model expects, since it was trained with audio converted by FFmpeg.)
7.  If the model categorizes the audio clip as "wake word said", then act accordingly.

You could also skip steps 2 and 3, that involve using VAD, and continually trim audio clips to make inference with the model, but this option is more expensive in terms of computational resources.

## Javascript example

You can find an example implementation of continuous detection at `web_demo/`, which you can also try online at: [https://crafywakeword2.netlify.app/](https://crafywakeword2.netlify.app/)

This example uses [onnxruntime-web](https://github.com/microsoft/onnxruntime) to run the model, [vad-web](https://github.com/ricky0123/vad) for Voice Activity Detection, and [ffmpeg-wasm](https://github.com/ffmpegwasm/ffmpeg.wasm) for FFmpeg.

1.  Load the model, FFmpeg and activate the VAD.
2.  Records an audio clip when it detects the user's voice.
3.  Convert and normalize the audio clip using FFmpeg.
4.  Perform an inference to the model with the audio clip.
5.  If the model detects the wake word, it displays a message on the screen.

**Warning of this demo:** The longer you try this demo without reloading the page, the more RAM it will consume, until it causes a memory error. This is a FFmpeg-wasm issue, not CrafyWakeWord2. One way to fix this is to reload the FFmpeg-wasm session every few iterations, but we haven't added that to this page because it's just a demo.