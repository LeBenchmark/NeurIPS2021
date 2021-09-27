#!/bin/bash
#Date: 16/12/2020
#Author: Solène Evain

#Script pour vérifier si fichier wav existe dans MLS
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
#        for audio in *
#            do
#            name=$(basename $audio .wav)
            if [ -n "$(find . -maxdepth 1 -name '*.wav')" ] 
                then
                echo "Mon fichier existe"
            else
                echo "Mon fichier n'existe pas"
            fi
#            done
#       ls
        cd ..
        done
    cd ..
    done

