from asr import print_transcript, run as run_asr
from mt import Nllb200
from tts import Tts

device = 'cuda'

languages = [
    'en',
    'hi'
]

mt_model = Nllb200(device=device)
tts_model = Tts(device=device)

mt_model.load_model()

# TODO handle commited vs non-commited text

def handle_output(transcript: str, start_timestamp: float, end_timestamp: float, language: str, now: float):
    print_transcript(transcript, start_timestamp, end_timestamp, language, now)
    try:
        lang_idx = languages.index(language)
        src = language
        tgt = languages[1-lang_idx]
        translated = mt_model.translate(transcript, source=src, target=tgt)
        print_transcript(translated, start_timestamp, end_timestamp, tgt, now)
    except:
        print('skipping different language')

run_asr(handle_output)
