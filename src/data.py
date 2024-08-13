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
    
    def read_parallel(self, src_path_train:str, tgt_path_train:str, src_path_test:str, tgt_path_test:str, test_split:int=0.2) -> DatasetDict:
        '''
        Reads parallel data that is aligned line by line and turns it into a dataset.
        The data is preprocessed to not contain newlines or sentences that are identical in both languages.
        
        Args:
            src_path_train: the path to the file with the training data in the source language 
            tgt_path_train: the path to the file with the training data in the target language 
            src_path_test: the path to the file with the test data in the source language 
            tgt_path_test: the path to the file with the test data in the target language 
            test_split: percentage how much of the train/test split should be test data
        
        Returns:
            DatasetDict: the parallel dataset containing the text in source and target language
        '''
        # Read text into dataframe
        src_file_train = open(src_path_train, 'r')
        src_train = src_file_train.readlines()
        tgt_file_train = open(tgt_path_train, 'r')
        tgt_train = tgt_file_train.readlines()
        src_file_test = open(src_path_test, 'r')
        src_test = src_file_test.readlines()
        tgt_file_test = open(tgt_path_test, 'r')
        tgt_test = tgt_file_test.readlines()
        
        dataframe_train = self.preprocess(src_train, tgt_train)
        dataframe_test = self.preprocess(src_test, tgt_test)
        
        # Save dataframe
        filename_train = 'pickle_train.p'
        dataframe_train.to_pickle(filename_train)
        
        filename_test = 'pickle_test.p'
        dataframe_test.to_pickle(filename_test)
        
        # Load dataframe as dataset
        data_train = load_dataset('pandas', data_files=filename_train)
        data_test = load_dataset('pandas', data_files=filename_test)
        
        data_train = data_train["train"].train_test_split(test_size=test_split)

        # We do this because the test set is way too large otherwise and manual evaluation will be unfeasible
        # Feel free to remove this line later on when you have the official (smaller) test set
        data_test = data_test["train"].train_test_split(test_size=0.9)
        
        print(data_train["train"][0:10])

        data = DatasetDict({
            'train': data_train['train'],
            'valid': data_train['test'],
            'test': data_test['train']
        })
        
        print(data)
        
        return data
    
    def preprocess(self, src:list[str], tgt:list[str]) -> pd.DataFrame:
        """
        Remove newlines, remove already translated sentences and turn the data into the correct format.
        
        Args:
            src: a list of the text in the source language
            tgt: a list of the text in the target language
            
        Returns:
            pandas.Dataframe: a Dataframe with the data in the correct format
        """
        
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
        
        return dataframe
    
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