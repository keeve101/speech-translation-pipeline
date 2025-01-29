from datasets import load_dataset, Dataset
from datasets.features import audio
from pandas import DataFrame
from common import STORAGE_DIR_DATA_FLEURS, STORAGE_DIR_CONVERSATION_DATA, STORAGE_DIR_REDUCED_FLEURS
from os import path, makedirs
import soundfile as sf
import numpy as np
import glob
from tqdm import tqdm

def read_audio_file(file_path: str):
    segments: list[str] = file_path.split(path.sep)
    segments.insert(-1, '*')
    for fp in glob.iglob(path.sep.join(segments)):
        return sf.read(fp)[0]
    print("can't find file: ", file_path)


def concatenate_audio_files(file_paths, output_path):
    """
    Concatenate audio files to a single 16kHz PCM WAV file.
    
    Parameters:
    file_paths (list): List of audio file paths to concatenate
    output_path (str): Path for output WAV file
    """
    # half a second of silence
    silence = np.zeros(8000, dtype=np.float32)

    audio_data = []
    # Read all audio files
    for filepath in file_paths:
        audio_data.append(read_audio_file(filepath))
        audio_data.append(silence.copy())

    # Concatenate audio data
    combined_audio = np.concatenate(audio_data)
    
    # Write to 16kHz PCM WAV
    sf.write(output_path, combined_audio, 16000, subtype='PCM_16')

def deduplicate_by_id(df):
    """
    Remove duplicate rows and set ID as index
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: Deduplicated DataFrame with ID as index
    """
    return df.drop_duplicates(subset='id').set_index('id')

def load_fleurs_with_path(lang) -> Dataset:
    return load_dataset(
        "google/fleurs",
        name=lang_id_to_fleurs[lang],
        cache_dir=STORAGE_DIR_DATA_FLEURS,
        trust_remote_code=True,
        split="train+validation+test",
    ).select_columns(['id', 'raw_transcription', 'path'])


lang_id_to_fleurs = {
    'en': 'en_us',
    'zh': 'cmn_hans_cn',
    'ms': 'ms_my',
    'id': 'id_id',
    'th': 'th_th',
    'hi': 'hi_in',
    'vi': 'vi_vn'
}

fleurs_eng: Dataset = load_fleurs_with_path('en')
fleurs_df = deduplicate_by_id(fleurs_eng.to_pandas())

def aggr(entry) -> int:
    matches = fleurs_eng.filter(lambda x: x['raw_transcription'] == entry)
    if len(matches) == 0:
            return int(-1)
    return int(matches[0]['id'])


try:
    flores_df = Dataset.load_from_disk('/tmp/flores').to_pandas()
except:
    flores_df: DataFrame = load_dataset("gsarti/flores_101", "eng", split="devtest+dev")\
            .filter(lambda x: 'travel' in x['topic'].lower())\
            .sort('URL').to_pandas()
    flores_df['fleurs_id'] = flores_df['sentence'].apply(aggr)
    flores_df = flores_df[flores_df['fleurs_id'] != -1]

    Dataset.from_pandas(flores_df).save_to_disk('/tmp/flores')


SPLIT_CONFIG = [
    [],
    [0],
    [0, 1],
    [0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
]


def create_conversation_dataset(lang):
    """
    Prepares a "conversation" dataset between english and `lang`.
    """

    fleurs_lang = load_fleurs_with_path(lang)

    fleurs_ids = set(fleurs_lang['id'])
    lang_df = flores_df[flores_df['fleurs_id'].isin(fleurs_ids)]
    lang_df = lang_df.groupby('URL')\
            .agg({ 'sentence': list, 'fleurs_id': list, 'id': 'count' })\
            .rename(columns={ 'id': 'size' }).reset_index()

    audio_base = path.join(STORAGE_DIR_CONVERSATION_DATA,'audio',lang)

    fleurs_lang = deduplicate_by_id(fleurs_lang.to_pandas())

    new_rows = []

    langs = ['en', lang]
    dfs = [fleurs_df, fleurs_lang]

    for row in tqdm(lang_df.itertuples(), total=len(lang_df)):
        split = SPLIT_CONFIG[row.size]
        for j in range(2):
            row_id = len(new_rows)
            transcription = []
            translation = []
            audio_files = []
            for i in range(len(split)):
                dfid = (j+split[i])%2
                audio_files.append(dfs[dfid].loc[row.fleurs_id[i]]['path'])
                transcription.append(dfs[dfid].loc[row.fleurs_id[i]]['raw_transcription'])
                translation_id = 1-dfid
                translation.append((
                    dfs[translation_id].loc[row.fleurs_id[i]]['raw_transcription'],
                    langs[translation_id],
                ))

            audio_file_path = path.join(audio_base, f'{len(new_rows)}.wav')
            concatenate_audio_files(audio_files, audio_file_path)
            new_row.append()
            new_rows.append((row_id, transcription, translation, path.join('audio', lang, f'{len(new_rows)}.wav')))

    dataset = Dataset.from_pandas(DataFrame(new_rows, columns=['id', 'transcription', 'translation', 'path']))
    dataset.save_to_disk(path.join(STORAGE_DIR_CONVERSATION_DATA, lang))
    return dataset

def create_reduced_fleurs_dataset(lang):
    """
    Prepares a "conversation" dataset between english and `lang`.
    """

    num_eg = 100

    fleurs_lang = load_fleurs_with_path(lang)
    fleurs_lang = deduplicate_by_id(fleurs_lang.to_pandas())

    fleurs_ids = set(fleurs_df.index)
    lang_df = fleurs_lang[fleurs_lang.index.isin(fleurs_ids)]
    lang_df = lang_df.iloc[np.random.choice(len(lang_df), num_eg)]

    audio_base = path.join(STORAGE_DIR_REDUCED_FLEURS,'audio')

    new_rows = []
    
    langs = ['en', lang]
    dfs = [fleurs_df, fleurs_lang]

    for lang in langs:
        makedirs(path.join(audio_base, lang), exist_ok=True)

    cols = ['id']
    for lang in langs:
        cols.append(f'{lang}_transcription')
        cols.append(f'{lang}_audio_path')

    for row in tqdm(lang_df.itertuples(), total=len(lang_df)):
        new_row = [row.Index]
        for lang, df in zip(langs, dfs):
            df_row = df.loc[row.Index]
            new_row.append(df_row['raw_transcription'])
            audio_file_path = path.join(audio_base, lang, f'{row.Index}.wav')
            concatenate_audio_files([df_row['path']], audio_file_path)
            new_row.append(path.join('audio', lang, f'{row.Index}.wav'))
        new_rows.append(new_row)

    dataset = Dataset.from_pandas(DataFrame(new_rows, columns=cols))
    dataset.save_to_disk(path.join(STORAGE_DIR_REDUCED_FLEURS, lang))
    return dataset

langs = [ 'zh', 'ms', 'id', 'th', 'hi', 'vi', ]
d = []

for lang in langs:
    d.append(create_reduced_fleurs_dataset(lang))
