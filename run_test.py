import os
import sys
import torch
import jiwer
from fuzzywuzzy import fuzz
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

audio_file_path = "path_to_audio_file.wav"

def transcribe(model_name, audio_file_path):
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

    audio_input = torch.tensor([tokenizer(audio_file_path, return_tensors="pt").input_values])

    with torch.no_grad():
        logits = model(audio_input).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    return transcription[0]

transcription_large_v2 = transcribe("openai/whisper-large-v2", audio_file_path)
transcription_medium = transcribe("openai/whisper-medium", audio_file_path)
transcription_small = transcribe("openai/whisper-small", audio_file_path)
transcription_tiny = transcribe("openai/whisper-tiny", audio_file_path)

def compare_transcriptions(reference, target):
    return fuzz.ratio(reference, target)

score_medium = compare_transcriptions(transcription_large_v2, transcription_medium)
score_small = compare_transcriptions(transcription_large_v2, transcription_small)
score_tiny = compare_transcriptions(transcription_large_v2, transcription_tiny)

print(f"fuzzy score: (Medium vs Large-v2): {score_medium}")
print(f"fuzzy score: (Small vs Large-v2): {score_small}")
print(f"fuzzy score: (Tiny vs Large-v2): {score_tiny}")

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation()
])

wer_medium = jiwer.wer(transcription_large_v2, transcription_medium, truth_transform=transformation, hypothesis_transform=transformation)
wer_small = jiwer.wer(transcription_large_v2, transcription_small, truth_transform=transformation, hypothesis_transform=transformation)
wer_tiny = jiwer.wer(transcription_large_v2, transcription_tiny, truth_transform=transformation, hypothesis_transform=transformation)

print(f"WER (Medium vs Large-v2): {wer_medium}")
print(f"WER (Small vs Large-v2): {wer_small}")
print(f"WER (Tiny vs Large-v2): {wer_tiny}")
