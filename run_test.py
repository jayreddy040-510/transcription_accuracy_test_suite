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

wer_medium = jiwer.wer(transcription_large_v2, transcription_medium, truth_transform=transformation, hypothesis_transform=transformation)
wer_small = jiwer.wer(transcription_large_v2, transcription_small, truth_transform=transformation, hypothesis_transform=transformation)
wer_tiny = jiwer.wer(transcription_large_v2, transcription_tiny, truth_transform=transformation, hypothesis_transform=transformation)

print(f"WER (Medium vs Large-v2): {wer_medium}", flush=True)
print(f"WER (Small vs Large-v2): {wer_small}", flush=True)
print(f"WER (Tiny vs Large-v2): {wer_tiny}", flush=True)
