#!/usr/bin/python3
"""
Date: 08/02/2021
Author: Marcely 

1. retrieves the metadata sentence information
2. load the corresponding audio
3. slice the audio into sentences 
4. updates the corpus json file
"""

import glob, argparse
import re
import xml.dom.minidom
from utils import *
import json

def load_speakers(content):
    speakers = content.getElementsByTagName("speaker")
    '''
    <speakers>
            <speaker name="S1" identity="" type="generic label" gender="M" generator="auto"/>
            <speaker name="S122" identity="" type="generic label" gender="F" generator="auto"/>
            <speaker name="S117" identity="" type="generic label" gender="F" generator="auto"/>
            ...
    </speakers>
    '''
    speakers_dict = dict()
    for speaker in speakers:
        spk_id = speaker.getAttribute("name")
        speakers_dict[spk_id] = dict()
        speakers_dict[spk_id]['gender'] = speaker.getAttribute("gender")
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

def load_xml(xml_file):
    content = xml.dom.minidom.parse(xml_file)
    speakers_dict = load_speakers(content)
    segment = content.getElementsByTagName("segment")
    '''
                    <segments>
                        <segment start="5.160000" end="8.040000" bandwidth="S" speaker="S1" generator="auto">
                                <text generator="auto"><sil/> <carillon/> c <sil/> <sil/> <end/> </text>
                                <graph id="0" type="1-best" generator="auto">
                                        <link id="0" start="0" end="1" type="filler" posterior="0.984"><sil/></link>
                                        <link id="1" start="1" end="2" type="filler" posterior="0.982"><carillon/></link>
                                        <link id="2" start="2" end="3" type="wtoken" posterior="0.497">c</link>
                                        <link id="3" start="3" end="4" type="filler" posterior="0.984"><sil/></link>
                                        <link id="4" start="4" end="5" type="filler" posterior="0.984"><sil/></link>
                                        <link id="5" start="5" end="6" type="filler" posterior="0.984"><end/></link>
                                </graph>
                        </segment>

    '''
    audios_dict = dict()
    index = 1
    for entry in segment:
        audios_dict[index] = dict()
        audios_dict[index]["spk_id"] = entry.getAttribute("speaker")
        audios_dict[index]["start"] = float(entry.getAttribute("start"))
        audios_dict[index]["end"] = float(entry.getAttribute("end"))
        audios_dict[index]["duration"] = audios_dict[index]["end"] - audios_dict[index]["start"]
        audios_dict[index]["textNode"] = retrieve_text(entry)
        index += 1
    return speakers_dict, audios_dict

def process(args):
    xml_files = glob.glob(args.metadata_folder + "/*xml")
    #json_data = dict()
    for xml_file in xml_files:
        try:
            speakers_dict, audio_dict = load_xml(xml_file)
            file_id = xml_file.split("/")[-1].replace(".transauto.xml","")
            audio = load_wav(args.wav_folder + "/" + file_id + ".wav")
            for sentence_id in audio_dict:
                #check if valid sentence +300s for partX with X >= 2
                not_first_part = re.search("_part[2-9]+", file_id)
                if not (not_first_part and audio_dict[sentence_id]["end"] <= 300):
                    #generate sliced audio
                    s_audio = slice_wav(audio_dict[sentence_id]["start"], audio_dict[sentence_id]["end"], audio)
                    #generate sentence id
                    s_id = file_id + "_s_" + str(sentence_id)
                    #write sliced audio
                    output_wav_file = args.output_folder + "/" + s_id + ".wav"
                    write_wav(output_wav_file, s_audio)
                    #update json file
                    trans = audio_dict[sentence_id]["textNode"]
                    duration = audio_dict[sentence_id]["duration"]
                    spk_id = audio_dict[sentence_id]["spk_id"]
                    spk_gender = speakers_dict[audio_dict[sentence_id]["spk_id"]]["gender"]
                    new_entry = create_json_entry(output_wav_file, trans, duration,spk_id,spk_gender)
                    write_json_entry(args.output_folder, args.json_file_name, s_id, new_entry)
        except Exception:
            with open("xml_problem_file","a") as dump_file:
                dump_file.write(xml_file + "\n")
    print("END")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-folder', help="Folder containing the files")
    parser.add_argument('--metadata-folder', help="Folder containing the metadata")
    parser.add_argument('--output-folder', help="Output folder for json files")
    parser.add_argument('--json-file-name', help="Output folder for json files")
    args=parser.parse_args()
    process(args)
