from copy import deepcopy
import numpy as np
from miditoolkit.midi import parser as mid_parser  
import miditoolkit as mtk
from utils import getNotes
def filterMelodyNotes(notes,method=2,threshold=20):
    """
    Filter out melody notes.
    Returns a tuple of (normal notes,melody notes)
    """
    # sort notes by start time and makes their lengths 1 tick
    notes=sorted(notes,key=lambda x:x.start)
    
    normal_notes=[]
    melody_notes=[]

    if method==2:
        
        # Construct reduced notes (Disregard note length)
        reduced_notes=deepcopy(notes)

        for note in reduced_notes:
            note.end=note.start+1

        reduced2notes={}

        for i,reduced_note in enumerate(reduced_notes):
            reduced2notes[reduced_note]=notes[i]

        for i,note in enumerate(reduced_notes):
            simultaneous_notes=notesBetween(note.start-threshold,note.start+threshold,reduced_notes)
            
            # Melody note has highest velocity
            if note.velocity==max([n.velocity for n in simultaneous_notes]):
                melody_notes.append(reduced2notes[note])
        
        normal_notes=[note for note in notes if not noteInNotes(note,melody_notes)]
    
    if method==3:  
        # find notes that have the same start time
        start2notes={}
        for note in notes:
            if note.start not in start2notes:
                start2notes[note.start]=[]
            start2notes[note.start].append(note)

        # For simultaneous notes, keep the one with highest velocity
        for start_time in start2notes:
            simultaneous_notes=start2notes[start_time]
            melody_notes.append(max(simultaneous_notes,key=lambda x:x.velocity))
        
        normal_notes=[note for note in notes if note not in melody_notes]

    

    # assert sum of melody and normal notes is equal to total notes
    checkEquivalence(notes,melody_notes,normal_notes)

    

    return normal_notes,melody_notes

def checkEquivalence(notes,melody_notes,normal_notes):
    """
    Check if melody notes and normal notes are equivalent to notes.
    """
    # deepcopy notes, melody_notes and normal_notes
    notes=deepcopy(notes)
    melody_notes=deepcopy(melody_notes)
    normal_notes=deepcopy(normal_notes)
    
    output_notes=melody_notes+normal_notes


    # sort notes by start time
    notes=sorted(notes,key=lambda x:(x.start,x.pitch))
    output_notes=sorted(output_notes,key=lambda x:(x.start,x.pitch))

    print(f"Percentage of melody notes: {int(100*len(melody_notes)/len(notes))}")
    # check if notes and output_notes are equivalent
    print(f"Number of notes: {len(notes)}")
    print(f"Number of output notes: {len(output_notes)}")
    assert len(notes)==len(output_notes)
    for i in range(len(notes)):
        assert notes[i].pitch==output_notes[i].pitch
        assert notes[i].start==output_notes[i].start
        assert notes[i].end==output_notes[i].end
        assert notes[i].velocity==output_notes[i].velocity
    print("Equivalence check passed")


def notesBetween(start_time,end_time,notes):
    """
    Get notes between start_time and end_time.
    """
    return [note for note in notes if (note.start>=start_time and note.end<=end_time) or (note.start<=start_time and note.end>=start_time)]

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
        
        z_threshold=1
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

    print(f"Number of normal notes: {len(normal_notes)}")
    print(f"Number of melody notes: {len(melody_notes)}")
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

def extractMelody(midi_file,output_filename="melody.mid",instrument=-1,threshold=2,save=True):
    """
    Extract melody from midi file.
    """
    midi_file=mtk.MidiFile(midi_file)
    notes=[]
    if instrument==-1:
        notes=getNotes(midi_file)
    else:
        notes=midi_file.instruments[instrument].notes

    notes=sorted(notes,key=lambda x:x.start)
    normal_notes,melody_notes=filterMelodyNotes(notes,method=3,threshold=threshold)
    out_midi=split2midi(normal_notes,melody_notes)
    if save:
        out_midi.dump(output_filename)
    return out_midi

    
heuristics={
    "velocity":velocityThreshold,
}

def noteInNotes(note,notes):
    """
    Returns True if note is in notes
    """
    for n in notes:
        if note.pitch==n.pitch and note.start==n.start and note.end==n.end:
            return True
    return False