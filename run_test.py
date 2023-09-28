import os
import sys
import torch
import jiwer
import pandas as pd
from fuzzywuzzy import fuzz
from transformers import pipeline

if "--path" not in sys.argv and "--p" not in sys.argv and "--d" not in sys.argv and "--dir" not in sys.argv:
    print("you need to specify a file with --p or --path or a directory with --d or --dir", flush=True)
    sys.exit(1)

audio_file_path, audio_dir = None, None

for idx, arg in enumerate(sys.argv):
    if arg == "--p" or arg == "--path":
        audio_file_path = sys.argv[idx + 1]
    if arg == "--d" or arg == "--dir":
        audio_dir = sys.argv[idx + 1]

def transcribe(model_name, audio_file_path):
    transcriber = pipeline("automatic-speech-recognition", model=model_name) 
    ret = transcriber(audio_file_path)
    return ret

def compare_transcriptions(reference, target):
    return fuzz.ratio(reference, target)

large_v2_transcriptions = []
fuzzy_tiny_scores = []
fuzzy_small_scores = []
fuzzy_medium_scores = []
fuzzy_tiny_sum, fuzzy_small_sum, fuzzy_medium_sum = 0,0,0


if audio_dir is not None:
    for filename in os.listdir(audio_dir):
        f = os.path.join(audio_dir, filename)
        if os.path.isfile(f):                        
            transcription_large_v2 = transcribe("openai/whisper-large-v2", f)
            transcription_medium = transcribe("openai/whisper-medium", f)
            transcription_small = transcribe("openai/whisper-small", f)
            transcription_tiny = transcribe("openai/whisper-tiny", f)
            
            score_medium = compare_transcriptions(transcription_large_v2, transcription_medium)
            score_small = compare_transcriptions(transcription_large_v2, transcription_small)
            score_tiny = compare_transcriptions(transcription_large_v2, transcription_tiny)
            fuzzy_tiny_sum += score_tiny
            fuzzy_small_sum += score_small
            fuzzy_medium_sum += score_medium

            large_v2_transcriptions.append(transcription_large_v2)
            fuzzy_tiny_scores.append(score_tiny)
            fuzzy_small_scores.append(score_small)
            fuzzy_medium_scores.append(score_medium)

print(f"tiny_avg: {fuzzy_tiny_sum/len(fuzzy_tiny_scores)}")
print(f"small_avg: {fuzzy_small_sum/len(fuzzy_small_scores)}")
print(f"medium_avg: {fuzzy_medium_sum/len(fuzzy_medium_scores)}")

d = {
        "large_v2_transcription": large_v2_transcriptions,
        "fuzzy_tiny_score": fuzzy_tiny_scores,
        "fuzzy_small_score": fuzzy_small_scores,
        "fuzzy_medium_score": fuzzy_medium_scores
    }

d2 = {
        "fuzzy_tiny_avg": [fuzzy_tiny_sum/len(fuzzy_tiny_scores)], 
        "fuzzy_small_avg": [fuzzy_small_sum/len(fuzzy_small_scores)], 
        "fuzzy_medium_avg": [fuzzy_medium_sum/len(fuzzy_medium_scores)] 
     }

df = pd.DataFrame(data=d)
df2 = pd.DataFrame(data=d2)
print(df)
print(df2)

if audio_file_path is not None:
    transcription_large_v2 = transcribe("openai/whisper-large-v2", audio_file_path)
    transcription_medium = transcribe("openai/whisper-medium", audio_file_path)
    transcription_small = transcribe("openai/whisper-small", audio_file_path)
    transcription_tiny = transcribe("openai/whisper-tiny", audio_file_path)

    print(f"transcription_large_v2: {transcription_large_v2}", flush=True)
    print(f"transcription_medium: {transcription_medium}", flush=True)
    print(f"transcription_small: {transcription_small}", flush=True)
    print(f"transcription_tiny: {transcription_tiny}", flush=True)


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

    transformed_large_v2 = transformation(transcription_large_v2['text'])
    transformed_medium = transformation(transcription_medium['text'])
    transformed_small = transformation(transcription_small['text'])
    transformed_tiny = transformation(transcription_tiny['text'])

    transformed_large_v2 = transformed_large_v2.split()
    transformed_medium = transformed_medium.split()
    transformed_small = transformed_small.split()
    transformed_tiny = transformed_tiny.split()

    wer_medium = jiwer.wer(transformed_large_v2, transformed_medium)
    wer_small = jiwer.wer(transformed_large_v2, transformed_small)
    wer_tiny = jiwer.wer(transformed_large_v2, transformed_tiny)

    print(f"WER (Medium vs Large-v2): {wer_medium}", flush=True)
    print(f"WER (Small vs Large-v2): {wer_small}", flush=True)
    print(f"WER (Tiny vs Large-v2): {wer_tiny}", flush=True)

