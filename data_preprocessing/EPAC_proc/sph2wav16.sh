#!/bin/bash


#audio folders root
root=$1

cd $root
module load sox

for d in $(ls -d */);
do
	cd $root$d
	#ls
	for f in *.sph; 
	do 
		sox -t sph "$f" -r 16000 -c 1 -b 16 -e signed-integer "${f%.*}.wav";
	done
	
done
