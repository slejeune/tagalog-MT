from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .googletrans import GoogleTranslate
    
from googletrans import Translator
from time import sleep
import tqdm
from datasets.dataset_dict import DatasetDict

class GoogleTranslate:
    
    def __init__(self, src_lang="tl", dest_code='en') -> None:
        self.translator = Translator()
        self.src_lang = src_lang
        self.dest_code = dest_code
        self.sleep_in_between_translations_seconds = 1
        self.long_sleep_in_between_translations_seconds = 60
        
    def translate(self, src:list) -> list:
        '''
        Translate the given dataset using Google Translate by translating line by line.
        
        Args:
            src: a list of strings in the source language
            
        Returns:
            list: a list of the predicted translations in the target language of the source strings
        '''
        
        translations = []
        for i in tqdm.tqdm(range(len(src))):
            # print("PROGRESS: " + str(i+1) + "/" + str(len(src)))
            
            try:
                if src[i] == '\n':
                    translations.append(src[i])
                else:
                    translation = self.translator.translate(src[i], src=self.src_lang).text
                    translations.append(translation)
                    self.__sleepBetweenQuery()
                
            except:
                self._longsleepBetweenQuery()
                translation = self.translator.translate(src[i], src=self.src_lang).text
                translations.append(translation)
                self.__sleepBetweenQuery()
        
            # print("RESULTS LENGTH: " + str(len(translations)))
        return translations
    
    def __sleepBetweenQuery(self):
        print('Sleeping for {}s after translation query...'.format(self.sleep_in_between_translations_seconds))
        sleep(self.sleep_in_between_translations_seconds)
    
    def _longsleepBetweenQuery(self):
        print('LONG SLEEP! Sleeping for {}s after translation query...'.format(self.long_sleep_in_between_translations_seconds))
        sleep(self.long_sleep_in_between_translations_seconds)