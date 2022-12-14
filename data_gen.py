import random
random.seed(420)
import pandas as pd
import pickle
from tqdm import tqdm
from utils import score2PerfFileMap
from IPython.display import display
import matplotlib.pyplot as plt
import os

def beat2TokenPosition(beat, beats, tokens,time_shift_positions=[]):
    beat_time=beats[beat]

    if len(time_shift_positions)==0:
        time_shift_positions=getTimeShifts(tokens)

    for i in range(len(time_shift_positions)):
        if time_shift_positions[i][1]>beat_time:
            return time_shift_positions[i-1][0]

def getTimeShifts(tokens):
    time_shift_positions=[]
    time_elapsed=0
    for i in range(len(tokens)):
        token=tokens[i]
        if token.type=="TIME_SHIFT":
            time_elapsed+=token.value/1000
            time_shift_positions.append((i, time_elapsed))
    return time_shift_positions

def splitTokens(tokens,beats,type="normal"):
    split_tokens=[]
    
    time_shift_positions=getTimeShifts(tokens)

    # find tokens between consequtive beats
    for i in range(len(beats)-1):
        start_beat=i
        end_beat=i+1
        start_token=beat2TokenPosition(start_beat, beats, tokens,time_shift_positions)
        end_token=beat2TokenPosition(end_beat, beats, tokens,time_shift_positions)
        split_tokens.append(tokens[start_token:end_token])
    
    return(split_tokens)
    

def read_annotations(annotations_file):
        annotations=pd.read_json(annotations_file).transpose()
        annotations['score_filename'] = annotations.index.map(lambda x: f"{'/'.join(x.split('/')[:-1])}/midi_score.mid")
        annotations.index=annotations.index.rename('performance_filename')
        annotations.reset_index(inplace=True)
        annotations = annotations[annotations['score_and_performance_aligned'] == True]
        # rename all columns containing midi_score to score
        for col in annotations.columns:
            if 'midi_score' in col:
                annotations.rename(columns={col: col.replace('midi_score', 'score')}, inplace=True)
        return annotations


def getBeats(annotations,filename,type="performance"):
    row=annotations[annotations[f"{type}_filename"]==filename]
    beats_type=row[f"{type}_beats_type"].iloc[0]
    beats=list(beats_type.keys())
    # convert to float
    beats = [float(x) for x in beats]
    return beats
 
def genData(pairs,score_dict,perf_dict,annotations,type="normal"):
    # iterate over train pairs and generate data
    data = {}
    for i in tqdm(range(len(pairs))):
        # read filenames
        score_filename=pairs[i][0]
        performance_filename=pairs[i][1]

        # get tokens
        score_tokens = score_dict[score_filename]
        performance_tokens = perf_dict[performance_filename]

        # get beats
        score_beats = getBeats(annotations,score_filename,"score")
        performance_beats = getBeats(annotations,performance_filename,"performance")

        # split tokens
        score_split = splitTokens(score_tokens, score_beats,type)
        performance_split=splitTokens(performance_tokens, performance_beats,type)

        # should be the same number of splits
        assert len(score_split)==len(performance_split) 
        
        # add to train data with key = performance_filename
        data[performance_filename] = (score_split, performance_split)
    return data

def splitPairs(s2p_map,train_split):
    train_pairs=[]
    test_pairs=[]

    for score_filename in s2p_map.keys():
        for performance_filename in s2p_map[score_filename]:
            # put train_split% in train and rest in test
            if random.random()<train_split:
                train_pairs.append((score_filename,performance_filename))
            else:
                test_pairs.append((score_filename,performance_filename))
    return train_pairs,test_pairs



from tqdm import tqdm
from midi_processing import mid2dat_anna
import pickle

# tokenise performances
def tokenise_performances(annotations,dataset_dir="Datasets/asap-dataset",method="normal"):
    token_dict={}
    for index in tqdm(annotations.index):
        row=annotations.loc[index]
        # get performance filename
        performance_filename = row['performance_filename']
        # get performance file
        performance_file = f"{dataset_dir}/{performance_filename}"
        # tokenise performance
        token_dict[performance_filename] = mid2dat_anna(performance_file)
    return token_dict

# tokenise scores
def tokenise_scores(annotations,dataset_dir="Datasets/asap-dataset",method="normal",min_beat=24):
    token_dict={}
    # get unique score filenames
    score_filenames = annotations['score_filename'].unique()
    for score_filename in tqdm(score_filenames):
        # get score filename
        score_filename = score_filename
        # get score file
        score_file = f"{dataset_dir}/{score_filename}"
        # tokenise score
        token_dict[score_filename] = mid2dat_anna(score_file)
    return token_dict

def tokenise(annotations,type="performance",dataset_dir="Datasets/asap-dataset",method="normal"):
    if type=="performance":
        return tokenise_performances(annotations,dataset_dir=dataset_dir,method=method)
    elif type=="score":
        return tokenise_scores(annotations,dataset_dir=dataset_dir,method=method)
    else:
        return None

def saveTokens(annotations,type="performance",method="normal",store_folder="Store/asap-dataset"):
    # tokenise and save pickle of scores
    token_dict = tokenise(annotations,type=type)
    with open(f'{store_folder}/{type}_dict_{method}.pickle', 'wb') as handle:
        pickle.dump(token_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)   

if __name__=="__main__":
    dataset_dir = "Datasets/asap-dataset"
    train_split=0.8
    store_dir="Store/asap-dataset"

    # read annotations
    print("Reading annotations..")
    annotations=read_annotations(f"{dataset_dir}/asap_annotations.json")

    # Check if store_dir exists
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    # Check if tokens exist 
    if not os.path.exists(f'{store_dir}/score_dict_normal.pickle'):
        print("Tokenising scores..")

        # tokenise scores
        saveTokens(annotations,type="score",method="normal",store_folder=store_dir)
        
        print("Tokenising performances..")
        # tokenise performances
        saveTokens(annotations,type="performance",method="normal",store_folder=store_dir)

    else:
        print("Tokens already exist")

    s2p_map=score2PerfFileMap(annotations)
    
    # split into train and test after shuffling
    print(f"Splitting pairs with {train_split*100}/{(1-train_split)*100} split")
    
    train_pairs,test_pairs=splitPairs(s2p_map,train_split)


    # read in tokens
    print("Reading tokens..")
    with open(f'{store_dir}/perf_dict_normal.pickle', 'rb') as handle:
        perf_dict = pickle.load(handle)
    with open(f'{store_dir}/score_dict_normal.pickle', 'rb') as handle:
        score_dict = pickle.load(handle)

    # save train data
    print("Generating training data..")
    train_data = genData(train_pairs,score_dict,perf_dict,annotations)
    with open(f'{store_dir}/train_data.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save test data
    print("Generating testing data..")
    test_data = genData(test_pairs,score_dict,perf_dict,annotations)
    with open(f'{store_dir}/test_data.pickle', 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data generated")