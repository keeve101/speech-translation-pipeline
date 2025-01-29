from functools import cache
from tqdm import tqdm
from asr import create_args, Runner
from datasets import Dataset
from common import STORAGE_DIR_REDUCED_FLEURS, STORAGE_DIR_RESULTS
from normalizers import BasicTextNormalizer, EnglishTextNormalizer
from os import path
from pathlib import Path
from itertools import permutations, product

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
batch_size = 25

output_folder = Path(STORAGE_DIR_RESULTS) / 'fleurs'
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
    'hi',
    'id',
    'ms',
    'vi',
    'th',
]

translation_settings = [0, 2]

configs = product(test_langs, translation_settings)

pipeline = CascadePipeline(['en'], device=device)
runner = Runner(pipeline)

args = create_args().parse_args()
args.log_level = 'INFO'
args.file = './jfk.wav'

runner.init(args)

print('Warming up the models')
runner.run()
pipeline.finish()

for lang, translation_setting in configs:
    print(f'### TESTING LANG: {lang} (translation: {translation_setting}) ###')

    pipeline.translation_setting = translation_setting
    pipeline.languages = ['en', lang]
    
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    lang_codes = ['en', lang]
    langs = '_'.join(lang_codes)

    dataset = Dataset.load_from_disk(path.join(STORAGE_DIR_REDUCED_FLEURS, lang))

    for i in tqdm(range(0, len(dataset), batch_size)):
        for src_lang in lang_codes:
            for tgt_lang in lang_codes:
                if src_lang == tgt_lang:
                    continue

                task = f"{src_lang}-{tgt_lang}-{translation_setting}"
                batch_output_file_path = (
                    output_folder / f"{task}_batch_{i}.json"
                )

                # To skip already processed batches
                if batch_output_file_path.exists():
                    continue

                batch = dataset.select(
                    list(range(i, min(i + batch_size, len(dataset))))
                )

                # Predict for batch
                predictions = []
                for sample in tqdm(batch, leave=False):
                    args.file = path.join(STORAGE_DIR_REDUCED_FLEURS, sample[f'{src_lang}_audio_path'])
                    runner.init(args)

                    latency = runner.run()
                    pipeline.finish()

                    src_txt = sample[f'{src_lang}_transcription']
                    tgt_txt = sample[f'{tgt_lang}_transcription']

                    translation = ''

                    for normalizer in get_normalizers(tgt_lang):
                        tgt_txt = normalizer(tgt_txt)

                    for text, lang in pipeline.translation_history:
                        for n in get_normalizers(lang):
                            text = n(text)
                        translation += text + '\n'


                    predictions.append(
                        {
                            "id": sample["id"],
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "source_ground_truth": src_txt,
                            "target_ground_truth": tgt_txt.strip(),
                            "prediction": translation.strip(),
                            "raw_transcription": pipeline.transcription_history,
                            "raw_prediction": pipeline.translation_history,
                            "latency": sum(latency)/len(latency)
                        }
                    )

                with open(batch_output_file_path, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=4)


    lang_combinations = ['-'.join(combo) for combo in permutations(lang_codes, 2)]
    all_predictions = []

    for lang_pattern in lang_combinations:
        batch_output_file_paths = glob.iglob(
            str(output_folder / f"*{lang_pattern}-{translation_setting}*batch*.json")
        )

        for file in batch_output_file_paths:
            with open(file, "r", encoding="utf-8") as f:
                batch_predictions = json.load(f)
                all_predictions.extend(batch_predictions)

    with open(
        f"{output_folder}/{langs}-{translation_setting}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_predictions, f, indent=4)

    evaluation_results = {
        "translation_setting": translation_setting,
    }

    # Evaluate the predictions
    for language in lang_codes:
        evaluation_results[language] = {}
        predictions_lang = [
            prediction for prediction in all_predictions if prediction["tgt_lang"] == language
        ]
        evaluation_results[language]['latency'] = sum(prediction["latency"] for prediction in predictions_lang)/len(predictions_lang)

        for metric in evaluation_metrics:
            predictions = [prediction["prediction"] for prediction in predictions_lang]
            references = [prediction["target_ground_truth"] for prediction in predictions_lang]

            kwargs = {}
            if metric == bleu_metric:
                kwargs['use_effective_order'] = True
            elif metric == chrf_metric:
                kwargs['word_order'] = 2

            results = metric.compute(predictions=predictions, references=references, **kwargs)

            # Store results in the dictionary under the corresponding language and metric
            evaluation_results[language][metric.name] = results

    # Write the results to a JSON file
    output_file_path = f"{output_folder}/{langs}-{translation_setting}_evaluation_results.json"
    with open(output_file_path, "w") as output_file:
       json.dump(evaluation_results, output_file, indent=4)
