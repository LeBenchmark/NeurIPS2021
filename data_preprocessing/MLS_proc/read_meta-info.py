#!/usr/bin/python3
"""
Date: 15/12/2020
Author: Solène

Lecture du fichier metainfo.txt pour récupérer le nombre d'heures d'enregistrement ainsi que le nombre d'hommes et de femmes 
Disposition des colonnes:  SPEAKER   |   GENDER   | PARTITION  |  MINUTES   |  BOOK ID   |             TITLE              |            CHAPTER """

import re
from datetime import timedelta
import time

meta_file=open(file="metainfo.txt",mode='r')
find_data=re.compile(r"(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\n")
male_num,female_num=0,0
total_duration=timedelta(seconds=0)
spk_memory=0

for line in meta_file:
    if find_data.match(line):
        found_line=find_data.match(line)

        ## récupération des groupes
        spk=found_line.group(1).strip()
        gender=found_line.group(2).strip()
        #partition=found_line.group(3).strip()
        minutes=found_line.group(4).strip()
        #book_id=found_line.group(5).strip()
        #title=found_line.group(6).strip()
        #chapter=found_line.group(7).strip()

        ## calcul du nombre d'hommes et de femmes
        if gender=="M":
            if spk!=spk_memory:
                print(spk,spk_memory)
                male_num+=1
                spk_memory=spk
        elif gender=="F" :
            if spk!=spk_memory:
                female_num+=1
                spk_memory=spk
        

        ## calcul du nombre global de minutes dans le corpus
        ## décomposer les durées qui sont du type "9.452" et les additionner
        if minutes != "MINUTES":
            duration=timedelta(minutes=float(minutes))
            #print("total_duration="+str(total_duration)+" new time="+str(duration))
            total_duration=total_duration+duration
print("Number of male voices: "+str(male_num))
print("Number of female voices: "+str(female_num))
print("total duration: "+str(total_duration))

         

