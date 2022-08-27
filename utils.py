import pandas as pd
from IPython.display import display


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

