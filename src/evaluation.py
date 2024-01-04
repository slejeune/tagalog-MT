from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .evaluation import Evaluation

import evaluate
import numpy as np
from transformers import NllbTokenizer

class Evaluation:
    
    def __init__(self, src:str="tgl_Latn", tgt:str="eng_Latn"):
        self.bleu = evaluate.load("sacrebleu")
        self.comet = evaluate.load('comet')
        
        self.tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src, tgt_lang=tgt)
    
    def eval(self, predictions:list, labels:list, source:list) -> dict:
        '''
        Evaluates predicted translations using the BLEU and COMET scores.
        
        Args:
            predictions:the predicted translation in the target language
            labels: the text in the target language
            source: the text before translation in the source language
            
        Returns:
            dict: the BLEU and COMET scores
        '''
        score = {}
        score['bleu'] = self.bleu.compute(predictions=predictions, references=labels)
        score['comet'] = self.comet.compute(predictions=predictions, references=labels, sources=source)
        return score
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.eval(predictions=predictions, references=labels)
    
    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = self.bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result