import os
import sys
import torch
import jiwer
from fuzzywuzzy import fuzz
from transformers import pipeline

if "--path" not in sys.argv and "--p" not in sys.argv:
    print("you need to specify a file with --p or --path", flush=True)
    sys.exit(1)

for idx, arg in enumerate(sys.argv):
    if arg == "--p" or arg == "--path":
        audio_file_path = sys.argv[idx + 1]

def transcribe(model_name, audio_file_path):
    transcriber = pipeline("automatic-speech-recognition", model=model_name) 
    ret = transcriber(audio_file_path)
    return ret

transcription_large_v2 = transcribe("openai/whisper-large-v2", audio_file_path)
transcription_medium = transcribe("openai/whisper-medium", audio_file_path)
transcription_small = transcribe("openai/whisper-small", audio_file_path)
transcription_tiny = transcribe("openai/whisper-tiny", audio_file_path)

print(f"transcription_large_v2: {transcription_large_v2}", flush=True)
print(f"transcription_medium: {transcription_medium}", flush=True)
print(f"transcription_small: {transcription_small}", flush=True)
print(f"transcription_tiny: {transcription_tiny}", flush=True)

def compare_transcriptions(reference, target):
    return fuzz.ratio(reference, target)

score_medium = compare_transcriptions(transcription_large_v2, transcription_medium)
score_small = compare_transcriptions(transcription_large_v2, transcription_small)
score_tiny = compare_transcriptions(transcription_large_v2, transcription_tiny)

print(f"fuzzy score: (Medium vs Large-v2): {score_medium}", flush=True)
print(f"fuzzy score: (Small vs Large-v2): {score_small}", flush=True)
print(f"fuzzy score: (Tiny vs Large-v2): {score_tiny}", flush=True)

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation()
])

# ... [previous code]

# Apply transformations
transformed_large_v2 = transformation(transcription_large_v2['text'])
transformed_medium = transformation(transcription_medium['text'])
transformed_small = transformation(transcription_small['text'])
transformed_tiny = transformation(transcription_tiny['text'])

# Split the transformed transcriptions into lists of words
transformed_large_v2 = transformed_large_v2.split()
transformed_medium = transformed_medium.split()
transformed_small = transformed_small.split()
transformed_tiny = transformed_tiny.split()

print(transformed_large_v2)
print(transformed_medium)
print(transformed_small)
print(transformed_tiny)

# Calculate WER
wer_medium = jiwer.wer(transformed_large_v2, transformed_medium)
wer_small = jiwer.wer(transformed_large_v2, transformed_small)
wer_tiny = jiwer.wer(transformed_large_v2, transformed_tiny)

print(f"WER (Medium vs Large-v2): {wer_medium}", flush=True)
print(f"WER (Small vs Large-v2): {wer_small}", flush=True)
print(f"WER (Tiny vs Large-v2): {wer_tiny}", flush=True)
