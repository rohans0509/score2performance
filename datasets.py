import random
import pickle
from re import L
from typing import NamedTuple
import torch
from torch.utils.data import Dataset
from collections import Counter
import random
import torch
import pickle
import random 
import pretty_midi as pm

import matplotlib.pyplot as plt
random.seed(42)

MASK_IDX = 390
PAD_IDX = 391

class MidiToken(NamedTuple):
    '''
    Used to store a MIDI event.
    Valid (type, value) pairs are as follows:
    NOTE_ON: [0,127]
    NOTE_OFF: [0,127]
    TIME_SHIFT: [10,1000] (goes by 10s)
    SET_VELOCITY: [0,124] (goes by 4s)
    START: [0]
    STOP: [0]
    MASK: [0]
    PADDING: [0]
    '''
    type: str
    value: int
    # t: int

def key_mapping(event):
    '''
    Given a MIDI event, return a unique index.
    '''
    if event.type == "NOTE_ON":
        return list(range(0,128))[event.value]
    if event.type == "NOTE_OFF":
        return list(range(128,256))[event.value]
    if event.type == "TIME_SHIFT":
        return list(range(256,356))[int(event.value // 10) - 1]
    if event.type == "SET_VELOCITY":
        return list(range(356,388))[int(event.value // 4)]
    if event.type == "START":
        return 388
    if event.type == "STOP":
        return 389
    if event.type == "MASK":
        return MASK_IDX
    if event.type == "PADDING":
        return PAD_IDX

def tok_mapping(token):
    '''
    Given a MIDI token index, return the associated MIDI event.
    '''
    if torch.is_tensor(token):
        token = token.item()
    if token >=0 and token < 128:
        return MidiToken("NOTE_ON", token)
    if token >= 128 and token < 256:
        return MidiToken("NOTE_OFF", token-128)
    if token >= 256 and token < 356:
        return MidiToken("TIME_SHIFT", ((token-256) * 10) + 10)
    if token >= 356 and token < 388:
        return MidiToken("SET_VELOCITY", (token-356) * 4)
    if token == 388:
        return MidiToken("START", 0)
    if token == 389:
        return MidiToken("STOP", 0)
    if token == MASK_IDX:
        return MidiToken("MASK", 0)
    if token == PAD_IDX:
        return MidiToken("PADDING", 0)



class PianoRollMidiToken(NamedTuple):
    '''
    Used to store a MIDI event.
    Valid (type, value) pairs are as follows:
    NOTE_ON: [0,127]
    NOTE_OFF: [0,127]
    TIME_SHIFT: [10,1000] (goes by 10s)
    SET_VELOCITY: [0,124] (goes by 4s)
    START: [0]
    STOP: [0]
    MASK: [0]
    PADDING: [0]
    '''
    type: str
    value: int

def key_mapping(event):
    '''
    Given a MIDI event, return a unique index.
    '''
    if event.type == "NOTE_ON":
        return list(range(0,128))[event.value]
    if event.type == "NOTE_OFF":
        return list(range(128,256))[event.value]
    if event.type == "TIME_SHIFT":
        return list(range(256,356))[int(event.value // 10) - 1]
    if event.type == "SET_VELOCITY":
        return list(range(356,388))[int(event.value // 4)]
    if event.type == "START":
        return 388
    if event.type == "STOP":
        return 389
    if event.type == "MASK":
        return MASK_IDX
    if event.type == "PADDING":
        return PAD_IDX

def tok_mapping(token):
    '''
    Given a MIDI token index, return the associated MIDI event.
    '''
    if torch.is_tensor(token):
        token = token.item()
    if token >=0 and token < 128:
        return MidiToken("NOTE_ON", token)
    if token >= 128 and token < 256:
        return MidiToken("NOTE_OFF", token-128)
    if token >= 256 and token < 356:
        return MidiToken("TIME_SHIFT", ((token-256) * 10) + 10)
    if token >= 356 and token < 388:
        return MidiToken("SET_VELOCITY", (token-356) * 4)
    if token == 388:
        return MidiToken("START", 0)
    if token == 389:
        return MidiToken("STOP", 0)
    if token == MASK_IDX:
        return MidiToken("MASK", 0)
    if token == PAD_IDX:
        return MidiToken("PADDING", 0)


class MidiPerformanceDataset(Dataset):
    '''
    MIDI sequences used in Music Transformer prepared for BERT-style training.
    Data augmentation includes pitch transpose and time stretch.
    The tokens are encoded as follows:
      0-127 = NOTE_ON
    128-255 = NOTE_OFF
    256-355 = TIME_SHIFT
    356-387 = SET_VELOCITY
    388-389 = START/STOP
        390 = MASK
        391 = PADDING
    '''
    def __init__(self, data_file, seq_len=512, mask_probs=[0.12, 0.015, 0.015],mode="score2perf"):
        self.seq_len = seq_len
        self.mask_probs = mask_probs

        self.data_file=data_file
        self.mode=mode


        # # read pickle data_file 
        print("Reading data from pickle file...")
        with open(data_file, 'rb') as f:
            self.data=pickle.load(f)
        print("Completed reading")
        
        for performance_filename,pair in self.data.items():
            score_token_list=pair[0]
            performance_token_list=pair[1]

            # insert Start tokens
            score_token_list[0].insert(0, MidiToken("START", 0))
            performance_token_list[0].insert(0, MidiToken("START", 0))

            # insert Stop tokens
            score_token_list[-1].append(MidiToken("STOP", 0))
            performance_token_list[-1].append(MidiToken("STOP", 0))

            # Update self.data
            self.data[performance_filename]=(score_token_list,performance_token_list)

        self.num_songs = len(self.data)
        self.beats = torch.tensor([len(song) for song in self.data.values()])
        self.files=list(self.data.keys())

    def augment(self, seq):
        # data augmentation: pitch transposition
        pitch_change = random.randint(-3,3)
        seq = [MidiToken(tok.type, tok.value + pitch_change) if tok.type in ("NOTE_ON", "NOTE_OFF") else tok for tok in seq]
        # data augmentation: time stretch
        time_stretch = random.choice([0.95, 0.975, 1.0, 1.025, 1.05])
        seq = [MidiToken(tok.type, int(min(max((((time_stretch * tok.value) + 5) // 10) * 10, 10), 1000))) if tok.type == "TIME_SHIFT" else tok for tok in seq]
        return seq

    def end_beat(self,pair, start_beat):

        score_token_list,performance_token_list=pair

        # convert to lists
        score_token_list=list(score_token_list)
        performance_token_list=list(performance_token_list)

        assert len(score_token_list)==len(performance_token_list)

        # try using performance_token_list and find the end beat where joined token list length exceeds seq_len
        length_performance_section=len(performance_token_list[start_beat])
        length_score_section=len(score_token_list[start_beat])

        end_beat=start_beat

        # return the end_beat such that none of the lengths of the performance and score sections are greater than seq_len
        while max(length_performance_section,length_score_section)<self.seq_len:
            end_beat+=1
            length_performance_section+=len(performance_token_list[end_beat])
            length_score_section+=len(score_token_list[end_beat])

            if end_beat==len(performance_token_list)-1:
                break
            
            
        return(end_beat-1)
            

    def get_batch(self, batch_size,show=False): 
        # chooses what songs we're going to use in this batch
        song_indices = torch.multinomial(torch.ones(self.num_songs), batch_size)

        # for each song in the batch, choose a random starting beat less than the length of the song
        start_beats = [random.randint(0, int(0.75*self.beats[song_index])) for song_index in song_indices]

        # iterate over song index pairs and construct end_beats
        end_beats = list(range(batch_size))

        for i in range(batch_size):
            file=self.files[song_indices[i]]
            pair=self.data[file]
            start_beat=start_beats[i]

            end_beats[i]=self.end_beat(pair,start_beat)

        # Get score and performance tokens using start_beats and end_beats
        score_tokens = []
        performance_tokens = []
        differences=[]
        for i in range(batch_size):
            # get sections
            file=self.files[song_indices[i]]
            score,performance=self.data[file]


            score_section=score[start_beats[i]:end_beats[i]+1]
            performance_section=performance[start_beats[i]:end_beats[i]+1]

           # join all sublists of score_section and performance_section
            score_section=sum(score_section,[])
            performance_section=sum(performance_section,[])

            # pad sections to seq_len if necessary
            if len(score_section)<self.seq_len:
                score_section=score_section+[MidiToken("PADDING",0)]*(self.seq_len-len(score_section))

            if len(performance_section)<self.seq_len:
                performance_section=performance_section+[MidiToken("PADDING",0)]*(self.seq_len-len(performance_section))

            # Clean sections
            score_section,performance_section=self.clean(score_section,performance_section)

            if show:
                midi_image(score_section)
                midi_image(performance_section)

            # compare score and performance sections
            diff=self.compare(score_section,performance_section)
            differences.append(diff)
            
            # Convert tokens to integers using key_mapping
            score_section=[key_mapping(tok) for tok in score_section]
            performance_section=[key_mapping(tok) for tok in performance_section]

            # Add to score_tokens and performance_tokens
            score_tokens.append(score_section)
            performance_tokens.append(performance_section)

        
        # Score tensor of shape (batch_size, seq_len)
        score_tensor = torch.tensor(score_tokens)
        # Performance tensor of shape (batch_size, seq_len)
        performance_tensor = torch.tensor(performance_tokens)
        # Difference tensor of shape (batch_size,1)
        difference_tensor = torch.tensor(differences)

        if self.mode=="score2perf":
            # Construct Input
            inp = score_tensor
            
            # Construct Output
            tgt = performance_tensor
        elif self.mode=="perf2score":
            # Construct Input
            inp = performance_tensor
            
            # Construct Output
            tgt = score_tensor
        else:
            raise ValueError("Mode must be either score2perf or perf2score")

        # assert that shape of inp and tgt are (batch_size, seq_len)
        assert inp.shape == (batch_size, self.seq_len)
        assert tgt.shape == (batch_size, self.seq_len)

        return {
            "inp": inp,
            "tgt": tgt,
            "diff": difference_tensor
        }
    
    def compare(self,score_section,performance_section):
        # loop through score_section and make a list of midi tokens of type NOTE_ON
        score_note_on_list=[]
        for tok in score_section:
            if tok.type=="NOTE_ON":
                score_note_on_list.append(tok.value)
        # loop through performance_section and make a list of midi tokens of type NOTE_ON
        performance_note_on_list=[]
        for tok in performance_section:
            if tok.type=="NOTE_ON":
                performance_note_on_list.append(tok.value)

        # sort both lists
        score_note_on_list.sort()
        performance_note_on_list.sort()

        # get counts of elements in both lists
        score_note_on_counts=Counter(score_note_on_list)
        performance_note_on_counts=Counter(performance_note_on_list)

        # get the difference between the two lists both positive and negative
        diff1=score_note_on_counts-performance_note_on_counts
        diff2=performance_note_on_counts-score_note_on_counts

        diffs=diff1+diff2

        # return the sum of the absolute values of the differences
        diff=sum([abs(x) for x in diffs.values()])




        
        
        return diff,diff/len(score_section)*100
        
    def clean(self,score_section,performance_section):
        # check if note on tokens are same in both sections, if not delete

        # loop through score_section and make a list of midi tokens of type NOTE_ON
        score_note_on_list=[]
        for i,tok in enumerate(score_section):
            if tok.type=="NOTE_ON":
                score_note_on_list.append((tok.value,i))
        # loop through performance_section and make a list of midi tokens of type NOTE_ON
        performance_note_on_list=[]
        for i,tok in enumerate(performance_section):
            if tok.type=="NOTE_ON":
                performance_note_on_list.append((tok.value,i))
        
        counts=[]

        return (score_section,performance_section)

from utils import read_mid,getNotes

class MidiMelodyDataset(Dataset):
    def __init__(self,data_file, seq_len=512):
        self.seq_len = seq_len

        # read pickle data_file={"score_file":(score_token_list,melody_tokens_list)}
        print("Reading data from pickle file...")
        with open(data_file, 'rb') as f:
            self.data=pickle.load(f) 
        print("Completed reading")
        
        for score_filename,pair in self.data.items():
            score_token_list=pair[0]
            melody_tokens_list=pair[1]

            # insert Start tokens
            score_token_list[0].insert(0, MidiToken("START", 0))
            melody_tokens_list[0].insert(0, MidiToken("START", 0))

            # insert Stop tokens
            score_token_list[-1].append(MidiToken("STOP", 0))
            melody_tokens_list[-1].append(MidiToken("STOP", 0))

            # Update self.data
            self.data[score_filename]=(score_token_list,melody_tokens_list)

        self.num_songs = len(self.data)
        self.beats = torch.tensor([len(song) for song in self.data.values()])
        self.files=list(self.data.keys())
    
    def end_beat(self,pair, start_beat):

        score_token_list,melody_token_list=pair

        # convert to lists
        score_token_list=list(score_token_list)
        melody_token_list=list(melody_token_list)

        # Check number of beats same
        assert len(score_token_list)==len(melody_token_list)

        # try using melody_token_list and find the end beat where joined token list length exceeds seq_len
        length_score_section=len(score_token_list[start_beat])
        length_melody_section=len(melody_token_list[start_beat])

        end_beat=start_beat

        # return the end_beat such that none of the lengths of the melody and score sections are greater than seq_len
        while max(length_melody_section,length_score_section)<self.seq_len:
            end_beat+=1
            length_melody_section+=len(melody_token_list[end_beat])
            length_score_section+=len(score_token_list[end_beat])

            if end_beat==len(melody_token_list)-1:
                break  
            
        return(end_beat-1)

    def get_batch(self, batch_size,show=False): 
        # chooses what songs we're going to use in this batch
        song_indices = torch.multinomial(torch.ones(self.num_songs), batch_size)

        # for each song in the batch, choose a random starting beat less than the length of the song
        start_beats = [random.randint(0, int(0.75*self.beats[song_index])) for song_index in song_indices]

        # iterate over song index pairs and construct end_beats
        end_beats = list(range(batch_size))

        for i in range(batch_size):
            file=self.files[song_indices[i]]
            pair=self.data[file]
            start_beat=start_beats[i]

            end_beats[i]=self.end_beat(pair,start_beat)

        # Get score and melody tokens using start_beats and end_beats
        score_tokens = []
        melody_tokens = []

        for i in range(batch_size):
            # get sections
            file=self.files[song_indices[i]]
            score,melody=self.data[file]

            score_section=score[start_beats[i]:end_beats[i]+1]
            melody_section=melody[start_beats[i]:end_beats[i]+1]

           # join all sublists of score_section and melody_section
            score_section=sum(score_section,[])
            melody_section=sum(melody_section,[])

            # pad sections to seq_len if necessary
            if len(score_section)<self.seq_len:
                score_section=score_section+[MidiToken("PADDING",0)]*(self.seq_len-len(score_section))

            if len(melody_section)<self.seq_len:
                melody_section=melody_section+[MidiToken("PADDING",0)]*(self.seq_len-len(melody_section))

            if show:
                midi_image(score_section)
                midi_image(melody_section)

            # Convert tokens to integers using key_mapping
            score_section=[key_mapping(tok,"score") for tok in score_section]
            melody_section=[key_mapping(tok,"melody") for tok in melody_section]
            
            # Add to score_tokens and melody_tokens
            score_tokens.append(score_section)
            melody_tokens.append(melody_section)

        
        # Score tensor of shape (batch_size, seq_len)
        score_tensor = torch.tensor(score_tokens)
        # melody tensor of shape (batch_size, seq_len)
        melody_tensor = torch.tensor(melody_tokens)
       
        # Construct Input
        inp = score_tensor
        
        # Construct Output
        tgt = melody_tensor

        # assert that shape of inp and tgt are (batch_size, seq_len)
        assert inp.shape == (batch_size, self.seq_len)
        assert tgt.shape == (batch_size, self.seq_len)

        return {
            "inp": inp,
            "tgt": tgt,
        }

    def key_mapping(self,tok,mode="melody"):
        if mode=="melody":
            if tok.type=="NOTE_ON_MELODY":
                return 1
            elif tok.type=="NOTE_ON":
                return 0
            else:
                return -1
        elif mode=="score":
            if tok.type=="NOTE_ON":
                return tok.pitch
            elif tok.type=="NOTE_OFF":
                return tok.pitch+128
            elif tok.type=="START":
                return 256
            elif tok.type=="STOP":
                return 257
            elif tok.type=="PADDING":
                return 258
            else:
                raise ValueError("Invalid token type")
        else:
            raise ValueError("Invalid mode")
    
    


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


# function to midi image
def midi_image(tokens):
    # read in midi file
    midi_data=dat2mid_anna(tokens+[])
    # get piano roll
    piano_roll =  midi_data.get_piano_roll(fs=100) # shape=(pitch, timestep)
    # plot the piano roll with length of yaxis=xaxis
    
    plt.imshow(piano_roll, aspect='auto', origin='lower')
    plt.show()
