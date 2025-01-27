from functools import cache
from tqdm import tqdm
from asr import create_args, Runner
from datasets import Dataset
from common import STORAGE_DIR_CONVERSATION_DATA, STORAGE_DIR_RESULTS
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from os import path
from pathlib import Path

import json
import torch
import gc
import glob
import evaluate

from main import CascadePipeline

"""
To run predictions on batches using different models. 

Predictions are stored, and batches are concatenated at the end.
"""

def detect_chinese(text):
    has_chinese = any(
        "\u4E00" <= char <= "\u9FFF" or "\u3400" <= char <= "\u4DBF" for char in text
    )
    return has_chinese


def add_spaces_in_chinese(prediction):
    reconstruct = ""
    for i, char in enumerate(prediction):
        if detect_chinese(char):
            if i != len(predictions) - 1:
                reconstruct += char + " "
            else:
                reconstruct += char
        else:
            reconstruct += char

    return reconstruct

@cache
def get_normalizers(lang):
    normalizers = [
        BasicTextNormalizer() if lang != "en" else EnglishTextNormalizer()
    ]

    if lang == "zh":
        normalizers.append(add_spaces_in_chinese)

    return normalizers

# Define batch size
batch_size = 32

output_folder = Path(STORAGE_DIR_RESULTS)
output_folder.mkdir(exist_ok=True)

# Define models
device = "cuda"

# Define evaluation metrics
wer_metric = evaluate.load("wer")
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
evaluation_metrics = [wer_metric, bleu_metric, chrf_metric]

test_langs = [
    'zh',
    'id',
    'hi',
    'ms',
    'vi',
    'th',
]

translation_setting = 0

pipeline = CascadePipeline(['en'], device=device)
runner = Runner(pipeline)

args = create_args().parse_args()
args.log_level = 'INFO'
args.file = './jfk.wav'
runner.init(args)

evaluation_results = {
}

for lang in test_langs:
    print('### TESTING LANG:', lang, '###')

    pipeline.translation_setting = translation_setting
    pipeline.languages = ['en', lang]
    
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    dataset = Dataset.load_from_disk(path.join(STORAGE_DIR_CONVERSATION_DATA, lang))

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_output_file_path = output_folder / f'{lang}_batch_{i}.json'

        # To skip already processed batches
        if batch_output_file_path.exists():
            continue

        batch = dataset.select(
            list(range(i, min(i + batch_size, len(dataset))))
        )

        # Predict for batch
        predictions = []
        for sample in tqdm(batch, leave=False):
            args.audio_path = sample['path']
            runner.init(args)

            runner.run()
            pipeline.finish()

            truth = ""
            pred = ""

            for text, lang in sample['translation']:
                for n in get_normalizers(lang):
                    text = n(text)
                truth += text + '\n'

            for text, lang in pipeline.translation_history:
                for n in get_normalizers(lang):
                    text = n(text)
                pred += text + '\n'

            predictions.append(
                {
                    "id": sample["id"],
                    "ground_truth": truth.strip(),
                    "raw_transcription": sample['transcription']
                    "raw_ground_truth": sample['translation'],
                    "prediction": pred.strip(),
                    "raw_prediction": pipeline.translation_history,
                }
            )

        with open(batch_output_file_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4)

    # Combine the batch outputs into a single file
    batch_output_file_paths = glob.glob(
        str(output_folder / f"{lang}_batch_*.json")
    )

    all_predictions = []
    for file in batch_output_file_paths:
        with open(file, "r", encoding="utf-8") as f:
            batch_predictions = json.load(f)
            all_predictions.extend(batch_predictions)

    with open(
        f"{output_folder}/{lang}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_predictions, f, indent=4)

    # Evaluate the predictions
    evaluation_results[lang] = {}
    for metric in evaluation_metrics:
        predictions = [prediction["prediction"] for prediction in all_predictions]
        references = [prediction["ground_truth"] for prediction in all_predictions]

        kwargs = {}
        if metric == bleu_metric:
            kwargs['use_effective_order'] = True
        elif metric == chrf_metric:
           kwargs['word_order'] = 2

        results = metric.compute(predictions=predictions, references=references, **kwargs)

        # Store results in the dictionary under the corresponding language and metric
        evaluation_results[lang][metric.name] = results

# Write the results to a JSON file
output_file_path = (
    f"{output_folder}/evaluation_results.json"
)
with open(output_file_path, "w") as output_file:
    json.dump(evaluation_results, output_file, indent=4)
