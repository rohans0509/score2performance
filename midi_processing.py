import os
import pickle
import pandas as pd
import pretty_midi as pm
from pathlib import Path
from datasets import MidiToken

def dat2mid_anna(seq, fname="test.mid"):
    '''
    Given a sequence of MIDI events, write a MIDI file.

        Parameters:
            seq (list of MidiToken): Sequence of MIDI event objects
            fname (str): Output filename
        
        Returns:
            None
    '''
    assert seq is not None
    assert isinstance(seq[0], MidiToken)
    curr_notes = [-1] * 128 # -1=inactive, else val=start_time
    curr_time = 0.0
    curr_vel = 0
    midi_data = pm.PrettyMIDI()
    piano = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))
    seq.insert(0, MidiToken("SET_VELOCITY", 40))
        
    for event in seq:
        if event.type == "NOTE_ON":
            curr_notes[event.value] = curr_time
        elif event.type == "NOTE_OFF" and curr_notes[event.value] != -1:
            note = pm.Note(velocity=curr_vel, pitch=event.value, start=curr_notes[event.value], end=curr_time)
            piano.notes.append(note)
            curr_notes[event.value] = -1
        elif event.type == "TIME_SHIFT":
            curr_time += event.value / 1000
        elif event.type == "SET_VELOCITY":
            curr_vel = int(event.value)
        
    midi_data.instruments.append(piano)
    return(midi_data)



def mid2dat_anna(midi_path):
    '''
    Given a MIDI file, convert into a sequence of MIDI events.

        Parameters:
            midi_path (str/Path): Input MIDI filename
        
        Returns:
            arr (list): List of MidiToken event objects
    '''
    arr = []
    if not isinstance(midi_path, str):
        midi_path = midi_path.as_posix()
    midi_data = pm.PrettyMIDI(midi_path)
    x = midi_data.get_piano_roll(fs=100) # shape=(pitch, timestep)
    
    active_notes = [] # unended NOTE_ON pitches
    time_acc = -10 # track time passed (ms) since last TIME_SHIFT (start at -10 to offset first increment)
    curr_vel = 0 # last SET_VELOCITY value
    
    # Iterate over timesteps
    for t in range(x.shape[1]):
        time_acc += 10
        for p in range(x.shape[0]):
            # When a note starts
            if x[p,t] and p not in active_notes:
                active_notes.append(p)
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc))
                    time_acc = 0
                if (x[p,t]//4)*4 != curr_vel:
                    curr_vel = (x[p,t]//4)*4
                    arr.append(MidiToken("SET_VELOCITY", curr_vel))
                arr.append(MidiToken("NOTE_ON", p))
            # When a note ends
            elif not x[p,t] and p in active_notes:
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc))
                    time_acc = 0
                active_notes.remove(p)
                arr.append(MidiToken("NOTE_OFF", p))
        if time_acc == 1000:
            arr.append(MidiToken("TIME_SHIFT", 1000))
            time_acc = 0
    # Write final NOTE_OFFs
    if active_notes:
        time_acc += 10
        arr.append(MidiToken("TIME_SHIFT", time_acc))
        for p in active_notes:
            if p != -1:
                arr.append(MidiToken("NOTE_OFF", p)) 
    return arr


# This script generates the MAESTRO pickle file.
if __name__ == "__main__":

    dataset_dir = "..\maestro"

    df = pd.read_csv(os.path.join(dataset_dir, 'maestro-v2.0.0.csv'))
    train_songs = df.loc[df['split'] == 'train', 'midi_filename']
    valid_songs = df.loc[df['split'] == 'validation', 'midi_filename']
    test_songs = df.loc[df['split'] == 'test', 'midi_filename']
    
    train_output = []
    for i,s in enumerate(train_songs):
        print("Train: %4d - %s" % (i+1, s))
        entry = {}
        entry['data'] = mid2dat_anna(os.path.join(dataset_dir, s[5:]))
        entry['meta'] = df.loc[df['midi_filename'] == s].iloc[0].to_dict()
        train_output.append(entry)
    with open("maestro_train.pickle", 'wb') as f:
        pickle.dump(train_output, f, protocol=4)

    valid_output = []
    for i,s in enumerate(valid_songs):
        print("Validation: %4d - %s" % (i+1, s))
        entry = {}
        entry['data'] = mid2dat_anna(os.path.join(dataset_dir, s[5:]))
        entry['meta'] = df.loc[df['midi_filename'] == s].iloc[0].to_dict()
        valid_output.append(entry)
    with open("maestro_valid.pickle", 'wb') as f:
        pickle.dump(valid_output, f, protocol=4)
    
    test_output = []
    for i,s in enumerate(test_songs):
        print("Test: %4d - %s" % (i+1, s))
        entry = {}
        entry['data'] = mid2dat_anna(os.path.join(dataset_dir, s[5:]))
        entry['meta'] = df.loc[df['midi_filename'] == s].iloc[0].to_dict()
        test_output.append(entry)
    with open("maestro_test.pickle", 'wb') as f:
        pickle.dump(test_output, f, protocol=4)