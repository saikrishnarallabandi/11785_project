#!/bin/bash

# Source the kaldi path
fbank_folder=fbank

mkdir -p fbank_ascii/cleaned
mkdir -p fbank_ascii/raw

# Get the mfccs and accomodate in a single file
for file in ${fbank_folder}/*.ark
do
 fname=$(basename "$file" .ark)
 cat ${fbank_folder}/${fname}.scp | while read f
 do
   n=`echo "${f}" | cut -d ' ' -f 1`
   echo $f | copy-feats scp:- ark,t:- | add-deltas ark:- ark,t:- > fbank_ascii/raw/${n}.mfcc # | apply-cmvn scp:${mfcc_folder}/../cmvn_${lang}.scp ark:- ark,t:- > ../data/${lang}/raw/${n}.mfcc
   cat fbank_ascii/raw/${n}.mfcc | sed '/\[$/d' | sed 's/]//g' > fbank_ascii/cleaned/${n}.mfcc
 done
done 

