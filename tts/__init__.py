from .mms import MmsTts, model_list

class Tts:
    def __init__(self, device: str = "cpu"):
        self.models = {
            lang: MmsTts(lang, device=device) for lang in model_list
        }

    def get_model_name(self):
        return "mms-tts"

    def _get_model(self, lang: str):
        model = self.models.get(lang)

        if model == None:
            raise Exception("Unknown lang code: " + lang)

        return model

    def load_lang(self, lang: str = 'en'):
        return self._get_model(lang).load_model()

    def unload_lang(self, lang: str = 'en'):
        return self._get_model(lang).unload()

    def synthesize(self, text: str, lang: str = 'en'):
        return self._get_model(lang).synthesize(text)
    
