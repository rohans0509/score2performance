import os
import pickle
import pandas as pd
import pretty_midi as pm
from pathlib import Path
from datasets import MidiToken
from utils import noteInNotes
def dat2mid_anna(seq, fname="test.mid"):
    '''
    Given a sequence of MIDI events, write a MIDI file.

        Parameters:
            seq (list of MidiToken): Sequence of MIDI event objects
            fname (str): Output filename
        
        Returns:
            None
    '''
    # deep copy
    seq = [MidiToken(x.type, x.value) for x in seq]
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



def mid2dat_anna(midi_path,type="PerformanceRNN",min_beat=24,melody_notes=None):
    if melody_notes is not None:
        type="Melody"   
    if type=="PerformanceRNN":
        return performanceRNN(midi_path)
    elif type=="Melody":
        return melodyRep(midi_path)
    else:
        return pianoRoll(midi_path,min_beat)
    
def pianoRoll(midi_path,min_beat):
    '''
    Given a MIDI file, convert into a piano roll.

        Parameters:
            midi_path (str/Path): Input MIDI filename
        
        Returns:
            arr (np.array): Piano roll array
    '''
    if not isinstance(midi_path, str):
        midi_path = midi_path.as_posix()
    midi_data = pm.PrettyMIDI(midi_path)
    x = midi_data.get_piano_roll(fs=100) # shape=(pitch, timestep)
    return x

def performanceRNN(midi_path):
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

# check pretty midi piano roll for instrument 1
def melodyRep(midi_path):
    '''
    Given a MIDI file, convert into a sequence of MIDI events. If note is in melody_notes have a NOTE_ON_MELODY/NOTE_OFF_MELODY token

        Parameters:
            midi_path (str/Path): Input MIDI filename
        
        Returns:
            arr (list): List of MidiToken event objects
    '''
    arr = []
    if not isinstance(midi_path, str):
        midi_path = midi_path.as_posix()
    midi_data = pm.PrettyMIDI(midi_path)
    timestep=100
    x = midi_data.get_piano_roll(fs=timestep) # shape=(pitch, length/timestep)
    melody_pr=midi_data.instruments[0].get_piano_roll(fs=timestep)

    # set non-zero velocities to 80
    x[x>0]=80
    melody_pr[melody_pr>0]=80

    assert melody_pr.shape==x.shape
    
    active_notes = [] # unended NOTE_ON pitches
    time_acc = -10 # track time passed (ms) since last TIME_SHIFT (start at -10 to offset first increment)
    curr_vel = 0 # last SET_VELOCITY value 

    num_melody=0
    num_other=0
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
                if melody_pr[p,t]:
                    arr.append(MidiToken("NOTE_ON_MELODY", p))
                    num_melody+=1
                else:
                    arr.append(MidiToken("NOTE_ON", p))
                    num_other+=1
            # When a note ends
            elif not x[p,t] and p in active_notes:
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc))
                    time_acc = 0
                active_notes.remove(p)
                if melody_pr[p,t]:
                    arr.append(MidiToken("NOTE_OFF_MELODY", p))
                else:
                    arr.append(MidiToken("NOTE_OFF", p))
        if time_acc == 1000:
            arr.append(MidiToken("TIME_SHIFT", 1000))
            time_acc = 0
    # Write final NOTE_OFFs and NOTE_OFF_MELODYs
    if active_notes:
        time_acc += 10
        arr.append(MidiToken("TIME_SHIFT", time_acc))
        for p in active_notes:
            if p != -1:
                if melody_pr[p,t]:
                    arr.append(MidiToken("NOTE_OFF_MELODY", p))
                else:
                    arr.append(MidiToken("NOTE_OFF", p))
    print(num_melody)
    print(num_other+num_melody)
    return arr