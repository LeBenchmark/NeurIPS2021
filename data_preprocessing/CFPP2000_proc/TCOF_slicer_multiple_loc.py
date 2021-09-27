#!/usr/bin/python3
"""
Date: 23/02/2021
Based on Marcely's EPAC_slicer
Author: Marcely 
Adaptation: Solène Evain

1. retrieves the metadata sentence information
2. load the corresponding audio
3. slice the audio into sentences 
4. updates the corpus json file
"""

import glob, argparse
import xml.dom.minidom
#requires pip3 install pydub --user at jean-zay
from pydub import AudioSegment
import json
import re

trans=re.compile(r'^\s*<Trans .* (audio_filename="(.*?)") .*$')
speaker=re.compile(r'^\s*<Speaker .*$')
section=re.compile(r'^\s*<Section .*$')
turn=re.compile(r'^\s*<Turn.*(speaker="(.*?)").*$')
turn_multiple_loc=re.compile(r'^\s*<Turn speaker="(spk.*?)\s(spk.*?)".*$')
turn_endtime=re.compile(r'^\s*<Turn.*(endTime="(.*?)").*$')
sync=re.compile(r'^\s*<Sync (time="(.*?)").*$')
text=re.compile(r'^\s*[^<].*[^>][\n|$]')
who=re.compile(r'^\s*<Who nb="(.*)".*$')

#speaker=re.compile(r's')

spk_id=re.compile(r'id="(.*)"')
spk_name=re.compile(r'name="(.*)"')
gender=re.compile(r'type="(.*)"')
accent=re.compile(r'accent="(.*)"')
mother_tongue=re.compile(r'dialect="(.*)"')

audio=re.compile(r'.*\.wav')
speakers_dict=dict()

def load_wav(file_id, wav_folder):
    #print(file_id)
    return AudioSegment.from_wav(wav_folder + "/" + file_id + ".wav")

def slice_wav(start, end, audio):
    start *= 1000 #s -> ms
    end *= 1000
    return audio[start:end]

def write_wav(file_name, audio):
    #print(file_name, audio)
    return audio.export(file_name,format="wav")

def load_speakers(speaker_match,audioName):
    
    speaker_match=speaker_match.group(0).split(' ')
    #speakers_dict=dict()

    for item in speaker_match:
        if spk_id.match(item) and spk_id.match(item).group(1)!="":
            spkID=spk_id.match(item)
            spkID=audioName+"_"+spkID.group(1)
            speakers_dict[spkID]=dict()
            speakers_dict[spkID]['name']="unk"
            speakers_dict[spkID]['gender']="unk"
            speakers_dict[spkID]['motherTongue']="unk"
            speakers_dict[spkID]['accent']="unk"

        if spk_name.match(item) and spk_name.match(item).group(1)!="":
            spkName=spk_name.match(item)
            spkName=spkName.group(1)
            speakers_dict[spkID]['name']=spkName

        if gender.match(item) and gender.match(item).group(1)!="":
            spkGender=gender.match(item)
            spkGender=spkGender.group(1)
            speakers_dict[spkID]['gender']=spkGender

        if mother_tongue.match(item) and mother_tongue.match(item).group(1)!="":
            spkMotherTongue=mother_tongue.match(item)
            spkMotherTongue=spkMotherTongue.group(1)
            speakers_dict[spkID]['motherTongue']=spkMotherTongue

        if accent.match(item) and accent.match(item).group(1)!="":
            spkAccent=accent.match(item)
            spkAccent=spkAccent.group(1)
            speakers_dict[spkID]['accent']=spkAccent 
    
    # for key,value in speakers_dict.items():
    #     print("here we are")
    #     print(key,value)

    return speakers_dict

def retrieve_text(content):
    textNode = content.getElementsByTagName("text")[0] 
    trans = ""
    while textNode.firstChild:
        child = textNode.firstChild
        try:
            token = child.tagName
        except AttributeError:
            token = child.data
        if token not in ["start","sil","end"]:
            trans += token
        textNode.removeChild(child)
    while "  " in trans:
        trans = trans.replace("  "," ")
    if trans and trans[0] == " ":
        trans = trans[1:]
    if trans and trans[-1] == " ":
        trans = trans[:-1]
    return trans

def load_trs(trs_file):
    f=open(trs_file,'r')
    index=0
    audios_dict=dict()
    #audios_dict[index]=dict()
    firstSync=True
    timeSync=0.0
    on_hold_text_to_print=False #permet de concaténer les transcriptions séparées par un <Comment ou <Event
    turnNum=1
    firstTurn=True
    notFirstTurn=False
    ignore_turn=False
    no_previous_text=False

    for line in f:
        if trans.match(line): #récupérer le nom de wav
            audioName=trans.match(line).group(2)
            if audio.match(audioName):
                audioName=audioName[:-4]

        elif speaker.match(line):  #récupérer les speakers
            speaker_match=speaker.match(line)
            speakers_dict=load_speakers(speaker_match,audioName)    

        #elif turn_multiple_loc.match(line): #si on rencontre un tour de parole avec plusieurs locuteurs!

        elif turn_multiple_loc.match(line): # si on rencontre un tour de parole avec plusieurs locuteurs
            print(line)
            ignore_turn=True # on créé un booleen pour pouvoir ignorer les lignes qui suivent jusqu'à ce qu'on ait un nouveau turn

        elif turn.match(line): #récupérer dans le turn: le nom de spk + le temps de fin du turn
            ignore_turn=False
            #on récupère le temps de fin du turn car sinon on a un segment sans temps de fin à la fin du traitement du fichier trs
            if firstTurn==True:  # si on est dans le tout premier turn du trs
                spkTurn=turn.match(line) # on récupère l'id du spk
                spkTurn=spkTurn.group(2)
                #print(audioName)
                #print(spkTurn)
                #print("firstTurn")
                endTime=turn_endtime.match(line) # on récupère le temps de fin du turn
                endTime=endTime.group(2)
                firstTurn=False # pour la suite du doc, si on retombe sur un turn, c'est que ça n'est pas le premier turn 
                
            else:  # à partir du deuxième turn
                spkTurn_previous=spkTurn  #on garde en mémoire le spk du turn précédent jusqu'à ce qu'on l'écrive quand on arrivera à sync
                spkTurn=turn.match(line) # on récupère le spk du turn actuel
                spkTurn=spkTurn.group(2)
                #print('\n')
                #print(audioName)
                print("previous"+str(spkTurn_previous))
                print("spk"+str(spkTurn))
                #print('not first turn')
                endTime_previous=endTime # on garde en mémoire le tmps de fin du turn précédent
                endTime=turn_endtime.match(line) # on récupère le temps de fin du turn actuel
                endTime=endTime.group(2) 
                
                notFirstTurn=True #booléen qui dit qu'on est pas dans le premier turn / qu'on est à un nouveau tour de parole

        elif sync.match(line): # récupérer les temps de sync des transcriptions
            timeSync=sync.match(line)
            timeSync=float(timeSync.group(2))

            if ignore_turn==True:
                pass
                            
            elif firstSync ==True: # si on rencontre le tout premier sync
                print("sync - step 2")
                audios_dict[index]=dict() #on créé une clé de valeur index (=0)
                audios_dict[index]['start'] = timeSync # on attribue la valeur timeSync à start
            
            elif notFirstTurn==False and on_hold_text_to_print==False: #si on a un sync sans avoir rencontré de texte /et qu'on est pas dans un nouveau tour de parole--> oon réinitialise le start
            #<Sync time="0"/>
            #    <Comment desc="le bébé pleure, ils entrent dans le cabinet"/>
            #    <Sync time="7.73"/>
                print("sync - step 3")
                audios_dict[index]['start'] = timeSync
                #no_previous_text=True

            elif notFirstTurn==True: # si on est pas sur le tout premier sync, mais qu'on est passés à un nouveau tour de parole
                print("sync- step4")
                if on_hold_text_to_print==True:

                    # on écrit dans audios_dict tout ce qui est relatif au tour de parole précédent
                    audios_dict[index]['end']=float(endTime_previous)
                    audios_dict[index]["duration"] =audios_dict[index]["end"] - audios_dict[index]["start"]
                    audios_dict[index]["spk_id"] = audioName+"_"+spkTurn_previous
                
                    on_hold_text_to_print=False # on précise que les données du tour précédent ont été écrites
                
                    index+=1 #on passe à l'index suivant
                    audios_dict[index]=dict() # on initialise l'entrée relative à l'index

                    # on écrit dans audios_dict tout ce qui est relatif au tour de parole actuel
                    audios_dict[index]['start']= timeSync #permanent
                    audios_dict[index]['end']=float(endTime) #endTime temporaire équivalent à la fin du tour de parole
                    audios_dict[index]["duration"] =audios_dict[index]["end"] - audios_dict[index]["start"] # duration temporaire
                    audios_dict[index]["spk_id"] = audioName+"_"+spkTurn #permanent

                    notFirstTurn=False #indique qu'on a déjà pris en compte qu'on était passé par un nouveau turn
                elif on_hold_text_to_print==False:
                    #on_hold_text_to_print=False
                    #index+=1
                    #audios_dict[index]=dict()

                    audios_dict[index]['start']=timeSync
                    #audios_dict[index]['end']=float(endTime)
                    #audios_dict[index]['duration']= audios_dict[index]['end'] - audios_dict[index]['start']
                    #audios_dict[index]['spk_id']=audioName+'_'+spkTurn
                    notFirstTurn=False

            else: # si on est ni sur le tout premier sync, ni dans un nouveau tour de parole (= si un seul spk dans trs)
                #on écrit dans audios_dict ce qui est relatif à la transcription précédente 
                #(rappel: on écrit une fois qu'on arrive sur sync, car sync est le temps de début du nouveau texte ET le temps de fin de l'ancien texte)
                print("sync-step 5")
                audios_dict[index]['end']=timeSync # le nouveau timesync marque la fin du segment précédent #actualisation
                audios_dict[index]["duration"] =audios_dict[index]["end"] - audios_dict[index]["start"] # on calcule la durée #actualisation
                audios_dict[index]["spk_id"] = audioName+"_"+spkTurn # on rentre l'ID du locuteur

                on_hold_text_to_print=False #plus de texte à écrire
                
                index+=1 # on passe à l'index suivant = le segment suivant

                audios_dict[index]=dict() # on initialise l'index
                audios_dict[index]['start']= timeSync # permanent
                audios_dict[index]['end']=float(endTime) # on donne en durée de fin le temps de fin du turn (sera actualisé au besoin)
                audios_dict[index]["duration"] =audios_dict[index]["end"] - audios_dict[index]["start"] # on calcule une durée provisoire
                audios_dict[index]["spk_id"] = audioName+"_"+spkTurn # permanent 
            firstSync=False #on est déjà passé par un sync


        elif text.match(line):
            textTrans=text.match(line)
            textTrans=textTrans.group(0)

            if ignore_turn==True:
                pass

            elif on_hold_text_to_print==True: # si on a pas rencontré de <Sync ou <Turn après le précédent texte et qu'on rencontre à nouveau du texte
                #= deux transcriptions sur deux lignes différentes qui ont le même <Sync 
                audios_dict[index]["textNode"]=audios_dict[index]["textNode"]+' '+textTrans.strip() # on concatène les transcriptions du même spkID
                on_hold_text_to_print=True  # on précise le statut du texte (=non rentré dans le dico)

            else: #si on rencontre une transcription
                audios_dict[index]["textNode"]=textTrans.strip() #on capture la transcription
                on_hold_text_to_print=True # on précise le statut du texte (=non rentré dans le dico)
            
    audios_dict[index]['spk_id']=audioName+"_"+spkTurn
    audios_dict[index]['end']=float(endTime)
    audios_dict[index]['duration']=audios_dict[index]['end'] - audios_dict[index]['start'] 
            
    for key,value in audios_dict.items():
        print(key,value)

    return speakers_dict, audios_dict
    
def create_json_entry(wav_file_path, speakers_dict, audios_dict):
    d = dict()

    try:
        d["path"] = wav_file_path
        d["trans"] = audios_dict["textNode"]
        d["duration"] = audios_dict["duration"]
        d["spk_id"] = audios_dict["spk_id"]
        d["spk_gender"] = speakers_dict[audios_dict["spk_id"]]["gender"]
    except KeyError as e:
        with open("trs_problem_file","a") as dump_file:
            dump_file.write("file: "+wav_file_path)
            dump_file.write("key not found " +str(e)+ " in audios_dict or speakers_dict \n")

    #print(d)
    return d

def write_json_file(output_folder, json_file_name, s_id, new_entry):
    try:
        with open(json_file_name) as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = dict()

    json_data[s_id] = new_entry
    
    with open(output_folder + "/" + json_file_name, mode='a', encoding='utf-8') as output_file:
        json.dump(json_data, output_file, ensure_ascii=False, indent=2, separators=(',', ': '))

def process(args):
    trs_files = glob.glob(args.metadata_folder + "/*trs")
    #json_data = dict()
    for trs_file in trs_files:
        #try:
        speakers_dict, audios_dict = load_trs(trs_file)
        file_id = trs_file.split("/")[-1].replace(".trs","")
        audio = load_wav(file_id, args.wav_folder)
        for sentence_id in audios_dict:
            #print(sentence_id)
            #generate sliced audio
            s_audio = slice_wav(audios_dict[sentence_id]["start"], audios_dict[sentence_id]["end"], audio)
            #generate sentence id
            s_id = file_id + "_s_" + str(sentence_id)
            #write sliced audio
            output_wav_file = args.output_folder  + s_id + ".wav"
            write_wav(output_wav_file, s_audio)
            #update json file
            new_entry = create_json_entry(output_wav_file, speakers_dict, audios_dict[sentence_id])
            write_json_file(args.output_folder, args.json_file_name, s_id, new_entry)
        #except Exception:
        #     with open("xml_problem_file","a") as dump_file:
        #         dump_file.write(trs_file + "\n")
        #print("END")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-folder', help="Folder containing the files")
    parser.add_argument('--metadata-folder', help="Folder containing the metadata")
    parser.add_argument('--output-folder', help="Output folder for json files")
    parser.add_argument('--json-file-name', help="Output folder for json files")
    args=parser.parse_args()
    process(args)
