function log_in_html(string) {
  var element = document.createElement('p');
  element.innerText = string;
  document.getElementById('htmlConsole').appendChild(element);
}

var myVAD;
var my_ffmpeg;
const { createFFmpeg } = FFmpeg;
var target_length = 16000 * (1700 / 1000);
var my_session;
var im_procesign_audio = false;

async function init_ffmpeg() {
  my_ffmpeg = createFFmpeg();
  await my_ffmpeg.load();
}

async function init_vad() {
  myVAD = await vad.MicVAD.new({
    positiveSpeechThreshold: 0.8,
    negativeSpeechThreshold: 0.45,
    redemptionFrames: 3,
    preSpeechPadFrames: 10,
    minSpeechFrames: 3,
    onSpeechStart: () => {
    },
    onVADMisfire: () => {
    },
    onSpeechEnd: (audio) => {
      // audio is float32Array in 16000 sampleRate
      on_get_audio(audio);
    }
  });
  myVAD.start();
}

async function loadModel(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Error al descargar el modelo: ${response.statusText}`);
  }

  // Paso 2: Convertir el contenido de la respuesta en un ArrayBuffer
  const modelArrayBuffer = await response.arrayBuffer();

  // Paso 3: Crear una nueva sesión de inferencia y cargar el modelo en ella
  const session = await ort.InferenceSession.create(modelArrayBuffer);

  my_session = session;
}

async function on_get_audio(audio) {
  if (im_procesign_audio) {
    return false;
  }
  im_procesign_audio = true;
  var audioUint8Array = float32ToWav(audio, 16000);
  await filterProcessAudio(audio, audioUint8Array);
  im_procesign_audio = false;
}

async function filterProcessAudio(audio, arrayBuffer) {
  await processAudio(arrayBuffer);
}

async function processAudio(file) {
  var init_time = Date.now();

  // Convertir el archivo y remuestrear a 16 kHz usando ffmpeg
  my_ffmpeg.FS('writeFile', 'input.wav', file);
  await my_ffmpeg.run('-i', 'input.wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-f', 'wav', 'output.wav');
  const outputData = my_ffmpeg.FS('readFile', 'output.wav');
  file = null;

  // Liberar memoria
  await my_ffmpeg.FS('unlink', 'input.wav');
  await my_ffmpeg.FS('unlink', 'output.wav');

  // Decodificar el archivo WAV en el navegador
  const audioContext = new AudioContext({ sampleRate: 16000 });
  const audioBuffer = await audioContext.decodeAudioData(outputData.buffer);
  const samples = audioBuffer.getChannelData(0);
  await audioContext.close();

  var init_time_ai = Date.now();

  // Normalizar las muestras
  const maxAbs = Math.max(...samples.map(Math.abs));
  const normalizedSamples = samples.map(sample => sample / maxAbs);

  var finalSamples = padOrTruncate(normalizedSamples, target_length);

  console.log(finalSamples);

  console.log('Preprocessed audio load time:', (Date.now() - init_time));

  var processedAudio = new ort.Tensor('float32', finalSamples, [1, target_length]);
  const logits = await runInference(my_session, processedAudio);
  const predictedClassId = getPredictedClassId(logits);

  console.log('predictedClassId', predictedClassId, '- load time:', (Date.now() - init_time), (Date.now() - init_time_ai));

  if (predictedClassId == 1) {
    log_in_html('Detected wake word! ' + obtenerFechaActual());
  } else {
    log_in_html('Not detected wake word. ' + obtenerFechaActual());
  }
}

function obtenerFechaActual() {
  const fecha = new Date();

  const año = fecha.getFullYear();
  const mes = String(fecha.getMonth() + 1).padStart(2, '0'); // getMonth() empieza en 0 (enero)
  const dia = String(fecha.getDate()).padStart(2, '0');

  const horas = String(fecha.getHours()).padStart(2, '0');
  const minutos = String(fecha.getMinutes()).padStart(2, '0');
  const segundos = String(fecha.getSeconds()).padStart(2, '0');

  return `${año}-${mes}-${dia} ${horas}:${minutos}:${segundos}`;
}

async function runInference(session, inputTensor) {
  // Realizar la inferencia
  const feeds = { [session.inputNames[0]]: inputTensor };
  const results = await session.run(feeds);
  return results[session.outputNames[0]];
}

function getPredictedClassId(logits) {
  // Obtener el índice de la clase con mayor probabilidad
  const array = logits.data;
  let maxIndex = 0;
  for (let i = 1; i < array.length; i++) {
    if (array[i] > array[maxIndex]) {
      maxIndex = i;
    }
  }
  return maxIndex;
}

function padOrTruncate(audio, targetLength) {
  if (audio.length < targetLength) {
    // Rellenar con ceros
    const paddedAudio = new Float32Array(targetLength);
    paddedAudio.set(audio);
    return paddedAudio;
  } else {
    // Truncar
    return audio.slice(0, targetLength);
  }
}

function float32ToWav(float32Array, sampleRate) {
  const numChannels = 1; // Mono audio
  const bytesPerSample = 2; // 16-bit audio

  // WAV Header
  const header = new Uint8Array(44);
  const view = new DataView(header.buffer);
  const dataLength = float32Array.length * bytesPerSample;
  const bufferLength = dataLength + 44;

  // RIFF chunk descriptor
  view.setUint32(0, 0x46464952, true); // 'RIFF'
  view.setUint32(4, bufferLength - 8, true); // File size - 8
  view.setUint32(8, 0x45564157, true); // 'WAVE'

  // fmt sub-chunk
  view.setUint32(12, 0x20746d66, true); // 'fmt '
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, numChannels, true); // NumChannels
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // ByteRate
  view.setUint16(32, numChannels * bytesPerSample, true); // BlockAlign
  view.setUint16(34, bytesPerSample * 8, true); // BitsPerSample

  // data sub-chunk
  view.setUint32(36, 0x61746164, true); // 'data'
  view.setUint32(40, dataLength, true); // Subchunk2Size

  // Convert Float32 to PCM 16-bit
  const output = new Uint8Array(bufferLength);
  output.set(header);

  const pcmData = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    pcmData[i] = Math.max(-32768, Math.min(32767, float32Array[i] * 32768));
  }

  new Int16Array(output.buffer, 44).set(pcmData);

  return output;
}

async function init_all() {
  await loadModel(window.location.origin + '/model.onnx');
  await init_ffmpeg();
  await init_vad();
}

document.addEventListener('DOMContentLoaded', function () {
  init_all();
});