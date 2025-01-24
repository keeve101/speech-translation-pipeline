from asr import run as run_asr, logger, TranscriptHandler
from asr import create_tokenizer, VACOnlineASRProcessor, OnlineASRProcessor

from mt import Nllb200
from tts import Tts

class CascadePipeline(TranscriptHandler):
    """
    Translation settings is an enum:
    == 0: translate only confirmed transcription as it comes
    == 1: translate per sentence, only translate confirmed transcription
    == 2: translate per sentence, translate confirmed and unconfirmed transcription

    Default is 0
    """
    def __init__(self, languages: list[str], translation_setting = 2, device='cuda'):
        super().__init__()
        self.languages = languages
        self.mt_model = Nllb200(device=device)
        self.mt_model.load_model()
        self.tts_model = Tts(device=device)
        self.translation_setting = translation_setting

        self.tokenizer = None
        if translation_setting != 0:
            self.tokenizer = create_tokenizer()

        self.last_transcribed_lang = None
        self.confirmed_transcription = ''
        self.unconfirmed_transcription = ''
        self.last_transcribed_sentence = ''
        self.confirmed_translation = ''
        self.unconfirmed_translation = ''

    def init(self, asr: OnlineASRProcessor):
        super().init(asr)
        # Disable TTS for now
        # for l in languages:
        #     self.tts_model.load_lang(l)

    def process_transcribed(self, transcript):
        self.confirmed_transcription += transcript

        online = self.asr.online if isinstance(self.asr, VACOnlineASRProcessor) else self.asr
        self.unconfirmed_transcription = online.to_flush(online.transcript_buffer.buffer)[2]

        logger.debug(' CFM: ' + self.confirmed_transcription)
        logger.debug('TODO: ' + self.unconfirmed_transcription)

    def process_translation(self, transcript: str, src: str, tgt: str):
        if src != self.last_transcribed_sentence:
            # Language has changed, translate whatever was confirmed transcribed but not confirmed
            # translated
            self.confirmed_translation += self.mt_model.translate(self.last_transcribed_sentence,source=src, target=tgt)
            # Reset the translation context
            self.last_transcribed_sentence = ''
            logger.debug(' CFM: ' + self.confirmed_translation)

        if self.tokenizer is None:
            self.confirmed_translation += self.mt_model.translate(transcript,source=src, target=tgt)
            logger.debug(' CFM: ' + self.confirmed_translation)
            return

        text = self.last_transcribed_sentence + transcript
        confirmed_len = len(text)

        if self.translation_setting == 2:
            text += self.unconfirmed_transcription

        sentences: list[str] = self.tokenizer.split(text, threshold=0.4)
        l = 0

        cfm_to_translate = ''
        uncfm_to_translate = ''
        self.last_transcribed_sentence = ''

        last_was_confirmed = True
        for i, s in enumerate(sentences):
            old_l = l
            l += len(s)
            if l <= confirmed_len and i+1 < len(sentences):
                cfm_to_translate += s
            else:
                if last_was_confirmed:
                    self.last_transcribed_sentence = s[:confirmed_len-old_l]
                uncfm_to_translate += s
                last_was_confirmed = False

        if len(cfm_to_translate) > 0:
            self.confirmed_translation += self.mt_model.translate(transcript,source=src, target=tgt)

        if len(uncfm_to_translate) > 0:
            self.unconfirmed_translation = self.mt_model.translate(uncfm_to_translate,source=src, target=tgt)

        logger.debug('TEXT: ' + text)
        logger.debug('SENT: ' + str(sentences))
        logger.debug(' CTR: ' + cfm_to_translate)
        logger.debug('UCTR: ' + uncfm_to_translate)
        logger.debug(' CFM: ' + self.confirmed_translation)
        logger.debug('TODO: ' + self.unconfirmed_translation)
        logger.debug('LAST: ' + self.last_transcribed_sentence)

    def handle(self, transcript: str, start_timestamp: float, end_timestamp: float, now: float):
        # TODO consider multiple languages; What about multiple languages for the online asr?
        src = self.asr.get_last_language()

        self.process_transcribed(transcript)
            
        try:
            lang_idx = self.languages.index(src)
            tgt = self.languages[1-lang_idx]

            self.process_translation(transcript, src, tgt)
            self.last_transcribed_sentence = src
        except ValueError:
            logger.debug(f"skipping different language {src}")

if __name__ == '__main__':
    pipeline = CascadePipeline(languages=['en', 'hi'])
    run_asr(pipeline)
    print(pipeline.confirmed_translation)
    print('[]', pipeline.unconfirmed_translation)
