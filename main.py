from src import *
from tqdm import tqdm
from datasets.dataset_dict import DatasetDict

def main():
    version = 'v2'
    
    create_train_test_split(version)
    
    # data = load_data(version)
    # save_txtfile(data, version)
    
    # nllbfinetuning(data)
    
    # pred_NLLB = nllb(data, finetuned=False)
    # pred_NLLB_finetuned = nllb(data, finetuned=True)
    # pred_GT = googletranslate(data)
    # pred_GT_manual = load_pred_txtfile('googletrans'+version+'.txt')
    
    # order_list = ["NLLB", "NLLB finetuned", "Google Translate auto", "Google Translate manual"]
    # pred = [pred_NLLB, pred_NLLB_finetuned, pred_GT, pred_GT_manual]
    
    # evaluate(data, pred, order_list=order_list)
    
def load_pred_txtfile(filename:str) -> list:
    '''
    Loads the predictions in the target language from an external txt file
    
    Returns:
        list: the list of the predictions in the target language
    '''
    with open(filename) as f:
        pred = f.readlines()
        
    for i in range(len(pred)):
            pred[i] = pred[i].replace("\n", "")
            
    return pred

def save_txtfile(data:DatasetDict, version:str) -> None:
    '''
    Save the test set of the source to a txt file.
    
    Args:
        data: the dataset the test set in the source language comes from
    '''
    with open("tagalog"+version+".txt","w") as wr:
        for line in data["test"]:
            wr.write(line['translation']['tg'] + '\n\n')

def create_train_test_split(version_name:str) -> None:
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
    data.save_train_test_split(parallel, version_name)

def load_data(version_name:str) -> DatasetDict:
    '''
    Loads the pickled train/test split created with 'create_train_test_split()'
    
    Returns:
        DatasetDict: the loaded parallel train/test split
    '''
    data = Data()
    return data.read_train_test_split(version_name)
    
def evaluate(parallel:DatasetDict, pred:list, single_sentence:bool=False, order_list:list=[]) -> None:
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
        print("pred: " + str(pred[0]))
        print("label: "+ parallel["test"][0]['translation']['en'])
        print(eval.eval([pred[0]], [parallel["test"][0]['translation']['en']], [parallel["test"][0]['translation']['tg']]))
    if type(pred[0])==list:
        with open("all_scores.txt","w") as wr:
            for i in range(len(pred)):
                evaluation = eval.eval(pred[i], labels, sources)
                wr.write('------------' + order_list[i] + '------------\n')
                wr.write('BLEU: ' + str(evaluation['bleu']['score']) + '\n')
                wr.write('COMET: ' + str(evaluation['comet']['mean_score']) + '\n')
                wr.write('Full eval: ' + str(evaluation))
                wr.write('\n\n')
    else:
        print(eval.eval(pred, labels, sources))
    
def nllb(parallel:DatasetDict, finetuned=False) -> list: 
    '''
    Generates the predicted translations using Meta's No Language Left Behind (NLLB) model.
    
    Args:
        parallel: the parallel dataset containing the text in source and target language
        finetuned: bool whether to use the finetuned version of the model or not
        
    Returns:
        list: the predicted translations in the target language
    '''
    translator = NLLBTranslator(src="tgl_Latn", tgt="eng_Latn", finetuned=finetuned)
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