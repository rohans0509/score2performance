import random
random.seed(420)
import pandas as pd
import pickle
from tqdm import tqdm
from midi_processing import mid2dat_anna
import matplotlib.pyplot as plt
import os
import mido
from utils import getNotes


def read_annotations(annotations_file):
        annotations=pd.read_json(annotations_file)
        
        # change score filename to performance filename
        annotations['score_filename'] = annotations['performance_filename']

        print(annotations.head())

        return annotations

def tokenize(annotations,type="score"):
    if type=="score":
        return tokenizeScore(annotations)
    elif type=="melody":
        return tokenizeMelodies(annotations)

def saveMelody(score_file,melody_instrument=0):
    # create melody file
    melody_file = f"{score_file.replace('_score.mid','_melody.mid')}"

    # read score
    score = mido.MidiFile(score_file)

    # create melody
    melody = mido.MidiFile()
    melody.ticks_per_beat = score.ticks_per_beat
    melody_track = mido.MidiTrack()
    melody.tracks.append(melody_track)

    # add notes to melody
    for msg in score.tracks[melody_instrument]:
        if msg.type == 'note_on':
            melody_track.append(msg)

    # save melody
    melody.save(melody_file)
    
def tokenizeMelodies(annotations):
    token_dict={}
    # get unique score filenames
    score_filenames = annotations['score_filename'].unique()
    for score_filename in tqdm(score_filenames):

        # get score filename
        score_filename = score_filename

        # get score file
        score_file = f"{dataset_dir}/{score_filename}"

        # replace .mid with _score.mid
        score_file = score_file.replace(".mid","_score.mid")

        melody_notes=getNotes(score_file,0)
        all_notes=getNotes(score_file)

        print(len(melody_notes))
        print(len(all_notes))

        # tokenize melody
        token_dict[score_filename] = mid2dat_anna(score_file,melody_notes=melody_notes)

    return token_dict

def tokenizeScore(annotations):
    token_dict={}
    # get unique score filenames
    score_filenames = annotations['score_filename'].unique()
    for score_filename in tqdm(score_filenames):
        # get score filename
        score_filename = score_filename
        # get score file
        score_file = f"{dataset_dir}/{score_filename}"
        # replace .mid with _score.mid
        score_file = score_file.replace(".mid","_score.mid")
        # tokenise score
        token_dict[score_filename] = mid2dat_anna(score_file)

    return token_dict

def saveTokens(annotations,store_folder="Store/Tokens/Score2Melody"):
    # tokenise and save pickle of scores and melodies
    token_dict_melody = tokenize(annotations,type="melody")
    with open(f"{store_folder}/melody_tokens.pkl", 'wb') as handle:
        pickle.dump(token_dict_melody, handle, protocol=pickle.HIGHEST_PROTOCOL)

    token_dict_score = tokenize(annotations,type="score")
    with open(f"{store_folder}/score_tokens.pkl", 'wb') as handle:
        pickle.dump(token_dict_score, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
    return (token_dict_score,token_dict_melody)

def loadTokens(store_folder="Store/Tokens/Score2Melody"):
    with open(f"{store_folder}/score_tokens.pkl", 'rb') as handle:
        score_tokens = pickle.load(handle)
    with open(f"{store_folder}/melody_tokens.pkl", 'rb') as handle:
        melody_tokens = pickle.load(handle)
    return score_tokens,melody_tokens


if __name__=="__main__":
    train_split=0.8
    dataset_dir="Store/Score2Melody"
    out_dir="Store/Tokens/Score2Melody"

    # read annotations
    print("Reading annotations..")
    annotations=read_annotations(f"{dataset_dir}/annotations.json")

    # Check if tokens exist 
    if not os.path.exists(f'{out_dir}"score_tokens.pkl'):
        # save tokens
        print("Tokenising and saving tokens..")

        score_tokens,melody_tokens=saveTokens(annotations,store_folder=out_dir)
    else:
        # load tokens
        print("Loading tokens..")
        score_tokens,melody_tokens=loadTokens(store_folder=out_dir)
    
    # split into train and test
    print("Splitting into train and test..")
    score_filenames = annotations['score_filename'].unique()
    random.shuffle(score_filenames)
    train_score_filenames = score_filenames[:int(len(score_filenames)*train_split)]
    test_score_filenames = score_filenames[int(len(score_filenames)*train_split):]

    # save train and test
    print("Saving train and test..")
    with open(f"{out_dir}/train_score_filenames.pkl", 'wb') as handle:
        pickle.dump(train_score_filenames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{out_dir}/test_score_filenames.pkl", 'wb') as handle:
        pickle.dump(test_score_filenames, handle, protocol=pickle.HIGHEST_PROTOCOL)

# For dataset class : After preprocessing don't need to dump midi back.