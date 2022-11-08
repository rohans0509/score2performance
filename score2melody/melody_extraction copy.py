from random import randrange
import numpy as np
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
import miditoolkit as mtk
from utils import *

def filterMelodyNotes(notes):
    """
    Filter out melody notes.
    Returns a tuple of (normal notes,ghost notes)
    """
    normal_notes=[]
    ghost_notes=[]
    for i,note in enumerate(notes):
        context=notes[i-10:i+10]
        if isMelodyNote(note,context):
            ghost_notes.append(note)
        else:
            normal_notes.append(note)
    return (normal_notes,ghost_notes)

def isMelodyNote(note,context_notes,heuristic="velocity",params=None):
    """
    Check if a note is a ghost note.
    """
    heuristic=heuristics[heuristic]
    return heuristic(note,context_notes,params)

def velocityThreshold(test_note,context_notes,params):
    ''' 
    Detect outliers in velocity.
    '''
    method=params["method"] if params is not None else 1
    all_velocities=[note.velocity for note in context_notes]+[test_note.velocity]
    outliers=getOutliers(all_velocities,method)

    # check if test_note is an outlier
    if test_note.velocity in outliers:
        return True

def getOutliers(data,method=1):
    if method==1:
        outliers=[]
        
        z_threshold=2
        median = np.median(data)
        std =np.std(data,)
        
        
        for y in data:
            z_score= (y - median)/std
            if z_score>z_threshold:
                outliers.append(y)
    elif method==2:
        data=sorted(data)
        q1, q3= np.percentile(data,[25,75])
        iqr = q3 - q1
        lower_bound = q1 -(1.5 * iqr) 
        outliers=[]
        for y in data:
            if y<lower_bound:
                outliers.append(y)
    return outliers

def split2midi(normal_notes,melody_notes):
    """
    Convert a list of notes to a midi file where first instrument is normal notes and second instrument is ghost notes.
    """
    normal_notes = [note for note in normal_notes if note.velocity != 0]
    melody_notes = [note for note in melody_notes if note.velocity != 0]
    mido_obj = mid_parser.MidiFile()
    beat_resol = mido_obj.ticks_per_beat

    # create instruments
    melody_instrument = mid_parser.Instrument(program=0)
    melody_instrument.name= "Melody Notes"

    normal_instrument = mid_parser.Instrument(program=1)
    normal_instrument.name= "Normal Notes"
    

    mido_obj.instruments.append(melody_instrument)
    mido_obj.instruments.append(normal_instrument)
    
    
    melody_instrument.notes = melody_notes
    normal_instrument.notes = normal_notes
    
    return mido_obj

def extractMelody(midi_file,output_filename="melody.mid",save=True):
    """
    Extract melody from midi file.
    """
    try:
        midi_file=mtk.MidiFile(midi_file)
    except:
        midi_file=midi_file

    notes=[]
    for instrument in midi_file.instruments:
        notes.extend(instrument.notes)
    notes=sorted(notes,key=lambda x:x.start)
    normal_notes,melody_notes=filterMelodyNotes(notes)
    out_midi=split2midi(normal_notes,melody_notes)
    if save:
        out_midi.dump(output_filename)
    return(out_midi)

    
heuristics={
    "velocity":velocityThreshold,
}