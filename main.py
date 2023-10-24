from src import *
from tqdm import tqdm
from datasets.dataset_dict import DatasetDict

def main():
    # create_train_test_split()
    
    data = load_data()
    
    # pred = googletranslate(data)
    pred = nllbbaseline(data)
    # pred = ["yup" for x in range(len(data["test"]))] # For checking if evaluation works
    
    # nllbfinetuning(data)
    
    evaluate(data, pred)

def create_train_test_split() -> None:
    '''
    Creates a train/test split and saves that split as a pickle.
    '''
    # paths.txt contains 2 lines of text:
    # 1. the path to the file containing the source language
    # 2. the path to the file containing the target language
    path_file = open("paths.txt", 'r')
    paths = path_file.read().splitlines()
    
    data = Data()
    parallel = data.read_parallel(paths[0],paths[1])
    data.save_train_test_split(parallel)

def load_data() -> DatasetDict:
    '''
    Loads the pickled train/test split created with 'create_train_test_split()'
    
    Returns:
        DatasetDict: the loaded parallel train/test split
    '''
    data = Data()
    return data.read_train_test_split()
    
def evaluate(parallel:DatasetDict, pred:list, single_sentence=False) -> None:
    '''
    Evaluate a prediction using the BLEU and COMET scores.
    
    Args:
        parallel: the parallel dataset containing the text in source and target language
        pred: the predicted translation
        single_sentence: whether we want to see a single sentence example
    '''
    eval = Evaluation()
    labels = [parallel["test"][i]['translation']['en'] for i in range(len(parallel['test']))]
    sources = [parallel["test"][i]['translation']['tg'] for i in range(len(parallel['test']))]
    
    if single_sentence:
        print("original: " + parallel["test"][0]['translation']['tg'])
        print("pred: " + pred[0])
        print("label: "+ parallel["test"][0]['translation']['en'])
        print(eval.eval([pred[0]], [parallel["test"][0]['translation']['en']], [parallel["test"][0]['translation']['tg']]))
    else:
        print(eval.eval(pred, labels, sources))
    
def nllbbaseline(parallel:DatasetDict) -> list: 
    '''
    Generates the predicted translations using Meta's No Language Left Behind (NLLB) model.
    
    Args:
        parallel: the parallel dataset containing the text in source and target language
        
    Returns:
        list: the predicted translations in the target language
    '''
    translator = NLLBTranslator(src="tgl_Latn", tgt="eng_Latn")
    test = [parallel["test"][i]['translation']['tg'] for i in range(len(parallel['test']))]

    print("Translating " + str(len(test)) + " sentence(s)")
    pred = list(tqdm(map(translator.translate, test),total=len(test))) # For a list of sentences
    
    return pred
    
def nllbfinetuning(parallel:DatasetDict) -> None:
    '''
    Finetunes Meta's No Language Left Behind (NLLB) model on the given data. The model is automatically saved to the directory.
    
    Args:
        parallel: the parallel dataset containing the text in source and target language
    '''
    translator = NLLBTranslator(src="tgl_Latn", tgt="eng_Latn")
    eval = Evaluation()
    translator.finetuning(parallel, eval)
    
def googletranslate(parallel:DatasetDict) -> list:
    '''
    Generates the predicted translations using Google's Google Translate.
    
    Args:
        parallel: the parallel dataset containing the text in source and target language
        
    Returns:
        list: the predicted translations in the target language
    '''
    translator = GoogleTranslate()
    test = [parallel["test"][i]['translation']['tg'] for i in range(len(parallel['test']))]
    print("Translating " + str(len(test)) + " sentence(s)")
    pred = translator.translate(test)   
    return pred
    
if __name__ == '__main__':
    main()