#!/bin/bash
#Date: 16/12/2020
#Author: Solène Evain

#Script pour changer les fichiers flac en wav avec sox pour le corpus MLS
#Il doit se trouver à la racine du corpus, là où se trouvent les dossiers train, dev et test

#Usage: $0 subset(train, dev ou test)"""

subset=$1

cd $subset/audio
for i in * #locuteur_ID
    do
    echo $i
    cd $i
    for j in * #book_ID
        do
        echo '\t' $j
        cd $j
        for audio in *
            do
            wav=$(basename $audio .wav)
            rm $wav.wav
        done
        for audio in *
            do
            name=$(basename $audio .flac)
            echo '\t \t'$audio
            sox $audio -r 16k -e signed-integer -b 16 -c 1 $name.wav
            done
        ls
        cd ..
        done
    cd ..
    done

