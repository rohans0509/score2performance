import pandas as pd
from IPython.display import display
import miditoolkit as mtk
import os
def read_annotations(path,process=True,only_aligned=True):
    """
    Reads the annotations from the given path.
    """
    # read in json file as pandas dataframe
    annotations=pd.read_json(path)

    if process:
        # add column for score_filename to annotations that converts row name to score_filename if that column does not exist
        if "score_filename" not in annotations.columns:
            annotations['score_filename'] = annotations.index.map(lambda x: f"{'/'.join(x.split('/')[:-1])}/midi_score.mid")
        # replace all column names containing midi_score with score if string name
        annotations.rename(columns=lambda x: x.replace('midi_score', 'score') if type(x)==str else x, inplace=True)

        # rename index to performance_filename
        annotations.index=annotations.index.rename('performance_filename')
        annotations.reset_index(inplace=True)

        
        if only_aligned:
            # only keep rows of annotations where score_and_performance_aligned is True	
            annotations = annotations[annotations['score_and_performance_aligned'] == True]
    
    return(annotations)

def read_mid(filename):
    """
    Reads a midi file and returns a midi object.
    """
    midi_obj=mtk.MidiFile(filename)
    return(midi_obj)

def getNotes(midi_obj,instrument=-1):
    if instrument==-1:
        notes=[]
        for instrument in midi_obj.instruments:
            notes.extend(instrument.notes)
        notes=sorted(notes,key=lambda x:x.start)
        return notes
    else:
        return midi_obj.instruments[instrument].notes

def getBeats(filename,annotations,type="score"):
    if type=="score":
        beats=annotations[annotations['score_filename']==filename]['score_beats'].iloc[0]
    elif type=="performance":
        beats=annotations[annotations['performance_filename']==filename]['performance_beats'].iloc[0]
    else:
        return None
    beats=[0]+beats
    return beats

def save_mid(midi_obj,filename):
    # create folders if they do not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    midi_obj.dump(filename)

def score2PerfFileMap(annotations):
    # construct a dataframe where first column is score_filename and other is list of all corresponding performances
    score_filenames=annotations['score_filename'].unique()
    score2perf_df=pd.DataFrame(columns=['score_filename','performance_filenames'])
    for i in range(len(score_filenames)):
        score_filename=score_filenames[i]
        performance_filenames=annotations[annotations['score_filename']==score_filename]['performance_filename'].unique()
        score2perf_df.loc[i]=[score_filename,performance_filenames]
    # convert to dictionary
    score2perf_dict=score2perf_df.set_index('score_filename')['performance_filenames'].to_dict()
    return score2perf_dict
    