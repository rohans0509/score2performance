import os
def extractScoreMelody(score_filename,performance_filename,annotations,dataset_dir="Sample_Data/asap-dataset"):

    score_path=os.path.join(dataset_dir,score_filename)
    performance_path=os.path.join(dataset_dir,performance_filename)

    # Read in score and performance
    score_obj=read_mid(score_path)
    performance_obj=read_mid(performance_path)

    # Get notes
    score_notes=getNotes(score_obj)
    performance_notes=getNotes(performance_obj)

    # Get beats
    score_beats = getBeats(score_filename,annotations,"score")
    performance_beats = getBeats(performance_filename,annotations,"performance")

    # Extract melody from performance and get melody notes
    performance_w_melody=extractMelody(performance_path,threshold=2)
    perf_melody_instrument=performance_w_melody.instruments[0]
    perf_melody_notes=perf_melody_instrument.notes

    # Get score melody
    score_melody_notes=set([])
    not_matched=0

    for i,perf_note in enumerate(perf_melody_notes):
        # Get the corresponding note in score
        score_note,sim_list=perfNote2ScoreNote(perf_note,score_obj,performance_obj,score_beats,performance_beats)
        if score_note is not None:
            score_melody_notes.add(score_note)
        else:
            not_matched+=1
        

    print("Not matched: ",not_matched)
    print("Not matched percentage",int(100*not_matched/len(perf_melody_notes)))
        
    
    # Normal notes are notes not in melody
    score_normal_notes=list(set(score_notes)-score_melody_notes)
    score_melody_notes=list(score_melody_notes)

    score_w_melody=split2midi(score_normal_notes,score_melody_notes)

    return score_w_melody,performance_w_melody

def getScoreWithVelocity(score_obj,performance_obj,annotations):
    # Get notes
    score_notes=getNotes(score_obj)
    performance_notes=getNotes(performance_obj)

    # Get beats
    score_beats = getBeats(score_filename,annotations,"score")
    performance_beats = getBeats(performance_filename,annotations,"performance")

    # Create a dictionary to map performance note to score note
    not_matched=0
    perf2score_dict={}
    for i,perf_note in tqdm(enumerate(performance_notes),total=len(performance_notes)):
        score_note,sim_list=perfNote2ScoreNote(perf_note,score_obj,performance_obj,score_beats,performance_beats)
        if score_note is not None:
            perf2score_dict[perf_note]=score_note
        else:
            not_matched+=1

    print("Not matched: ",not_matched)
    print("Not matched percentage",int(100*not_matched/len(performance_notes)))

    # Reverse perf2score_dict

    score2perf_dict={}

    for perf_note,score_note in perf2score_dict.items():
        if score_note not in score2perf_dict.keys():
            score2perf_dict[score_note]=[perf_note]
        else:
            score2perf_dict[score_note].append(perf_note)

    # For each score note, set the velocity to be the velocity of the corresponding performance note

    for score_note in score2perf_dict.keys():
        perf_note=score2perf_dict[score_note][0]
        score_note.velocity=perf_note.velocity

    # Remove instruments in score

    for i,track in enumerate(score_obj.instruments):
        if i!=0:
            score_obj.instruments.remove(track)

    # Add notes 
    for score_note in score2perf_dict.keys():
        score_obj.instruments[0].notes.append(score_note)

    return(score_obj)