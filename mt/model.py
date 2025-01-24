import torch
import gc
import time
from common import logger, STORAGE_DIR_MODEL

from transformers import (
    AutoModelForSeq2SeqLM,
)
from .tokenizer import NllbTokenizer

class Nllb200:
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", device: str = "cpu"):
        self.model_id = model_id
        self.tokenizers: NllbTokenizer= None
        self.model: AutoModelForSeq2SeqLM= None
        self.device = torch.device(device)

    def get_model_name(self):
        return self.model_id.split("/")[-1]

    def load_model(self):
        t = time.time()
        logger.info(f'Loading Nllb200 model ({self.get_model_name()})...')
        self.tokenizers = NllbTokenizer(self.model_id, cache_dir=STORAGE_DIR_MODEL + '/nllb')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, cache_dir=STORAGE_DIR_MODEL + '/nllb')
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

        self.model.to(self.device)
        return self.tokenizers, self.model

    def unload(self):
        if self.model is not None:
            del self.model
        if self.tokenizers is not None:
            del self.tokenizers
        # Flush the current model from memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def translate(self, text: str, source: str = "en", target: str = "en", max_length=1000):
        if self.tokenizers is None or self.model is None:
            self.load_model()

        tokenizer = self.tokenizers.get_tokenizer(source)

        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)

            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(self.tokenizers.get_lang_id(target)),
                max_length=max_length
            )
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translation
