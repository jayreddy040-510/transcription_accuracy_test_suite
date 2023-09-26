# Use a pipeline as a high-level helper
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

x = transcriber("/Users/jayreddy/asr_voice_filter/speaker_identification/Real-Time-Voice-Cloning/samples/1320_00000.mp3")

print(x)