from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .nllbtranslator import NLLBTranslator
    
from .evaluation import Evaluation
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamWeightDecay, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets.dataset_dict import DatasetDict

class NLLBTranslator:
    
    def __init__(self, src:str, tgt:str, version:str, finetuned:bool=False):
        self.src = src
        self.tgt = tgt
        self.version = version
        
        self.tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src, tgt_lang=tgt)
        
        if finetuned:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("finetuned_"+ version +"/")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        
        # self.checkpoint = "t5-small"
        self.checkpoint = "v3"
        # self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.checkpoint, return_tensors="tf")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.checkpoint, return_tensors="pt")
        
        self.optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
        
    def translate(self, src:list) -> list:
        """
        Generate string in the target language given the string in the source language.
        
        Args:
            src: list of strings in the source language
            
        Returns:
            list: the predicted translations in the target language
        """

        inputs = self.tokenizer(src, return_tensors="pt")
        
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt]
        )
        
        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        return translation
    
    def finetuning(self, parallel:DatasetDict, eval_class:Evaluation) -> None:
        '''
        Finetunes the NLLB translation model given a certain dataset.
        
        Args:
            parallel: the parallel dataset containing the text in source and target language
            eval_class: instance of the Evaluation class
        '''
            
        tokenized = parallel.map(function=self.preprocess_function, batched=True)
        
        batch_size = 8
        # Problem seems to be RAM??? maybe better after the other thing is done running
        
        model_name = self.checkpoint.split("/")[-1]
        
        args = Seq2SeqTrainingArguments(
            f"{model_name}-finetuned-{self.src}-to-{self.tgt}",
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True
        )
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["valid"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=eval_class.compute_metrics
        )
        
        print("START TRAINING...")
        trainer.train()
        print("TRAINING DONE")

        trainer.save_model("finetuned_"+ self.version + "/")
    
    def preprocess_function(self, examples:DatasetDict) -> DatasetDict:
        '''
        Tokenize the dataset and preprocess them for the model.
        
        Args:
            examples: the raw dataset
            
        Returns:
            DatasetDict: the tokenized and preprocessed dataset
        '''
        source_lang = "tg"
        target_lang = "en"
        prefix = "translate Tagalog to English: "
        
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        
        return model_inputs