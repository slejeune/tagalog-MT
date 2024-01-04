from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .data import Data
    
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import pandas as pd
import pickle 

class Data:
    
    def __init__(self):
        pass
    
    def read_parallel(self, src_path:str, tgt_path:str, test_split:int=0.2) -> DatasetDict:
        '''
        Reads parallel data that is aligned line by line and turns it into a dataset.
        The data is preprocessed to not contain newlines or sentences that are identical in both languages.
        
        Args:
            src_path: the path to the file with the source language
            tgt_path: the path to the file with the target language
            test_split: percentage how much of the train/test split should be test data
        
        Returns:
            DatasetDict: the parallel dataset containing the text in source and target language
        '''
        # Read text into dataframe
        src_file = open(src_path, 'r')
        src = src_file.readlines()
        
        tgt_file = open(tgt_path, 'r')
        tgt = tgt_file.readlines()
        
        # Remove all newlines
        for i in range(len(src)):
            src[i] = src[i].replace("\n", "")
        for i in range(len(tgt)):
            tgt[i] = tgt[i].replace("\n", "")
            
        while("" in src):
            src.remove("")
        while("" in tgt):
            tgt.remove("")
            
        # Remove sentences that are already translated
        duplicates = []
        for i in range(len(src)):
            if src[i] == tgt[i]:
                duplicates.append(i)
                
        src = [src[i] for i in range(len(src)) if i not in duplicates]
        tgt = [tgt[i] for i in range(len(tgt)) if i not in duplicates]
        
        assert len(src) == len(tgt)
        
        # { {'id':0, 'translation': {'eng':english, 'tgl':tagalog} }, {'id':1, 'translation': {'eng':english1, 'tgl':tagalog1 } }, ... }
        translation = [ {'tg':src[i], 'en':tgt[i]} for i in range(len(src)) ]
        id = [i for i in range(len(translation))]
        
        data = [id, translation]
        
        dataframe = pd.DataFrame(data).transpose()
        dataframe.columns =['id','translation']
        
        # Save dataframe
        filename = 'pickle.p'
        dataframe.to_pickle(filename)
        
        # Load dataframe as dataset
        data = load_dataset('pandas', data_files=filename)
        
        data = data["train"].train_test_split(test_size=test_split)
        
        data_train = data['train'].train_test_split(test_size=test_split)
        data_test = data_train['test'].train_test_split(test_size=0.5)

        data = DatasetDict({
            'train': data_train['train'],
            'valid': data_test['train'],
            'test': data_test['test']
        })
        
        return data
    
    def save_train_test_split(self, parallel:DatasetDict, version_name:str) -> None:
        '''
        Pickle the given dataset and save it to the directory.
        
        Args:
            parallel: the parallel dataset containing the text in source and target language
        '''
        with open(version_name+'.pkl', 'wb') as f:
            pickle.dump(parallel, f)
    
    def read_train_test_split(self, version_name:str) -> DatasetDict:
        '''
        Read the previously saved dataset from the directory.
        
        Returns:
            DatasetDict: the parallel dataset containing the text in source and target language
        '''
        with open(version_name+'.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict