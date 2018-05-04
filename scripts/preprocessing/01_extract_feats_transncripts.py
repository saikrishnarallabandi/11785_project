import librosa
import numpy as np
from scipy.io import wavfile as wf
import os,sys

# python 01_extract_feats_transncripts.py audio_folder transcript file feats_npy text_npy
# python 01_extract_feats_transncripts.py ../data/te-in-Train/Audios_1.1spedup ../data/te-in-Train/transcription.txt ../feats/feats40dim_train_1.1spedup.npy ../feats/transcripts_train_original.npy


outfile_feats = sys.argv[3]
outfile_transcripts = sys.argv[4]
wav_folder = sys.argv[1]
wav_files = sorted(os.listdir(wav_folder))
transcript_file = sys.argv[2]
f = open(transcript_file)
transcripts = []
for line in f:
    line = line.split('\n')[0]
    transcripts.append(' '.join(k for k in line.split()[1:]))

files = list(zip(wav_files,transcripts))

feats_array = []
transcripts_array = []
for i, (a,b) in enumerate(files):
   
   y,fs = librosa.load(wav_folder + '/' + a)
   #print("Shape of y is ", y.shape)
   mfcc = librosa.feature.mfcc(y=y,sr=float(fs), n_mfcc=40)
   mfcc = np.transpose(mfcc)
   feats_array.append(mfcc)
   transcripts_array.append(b)
   
   if i % 100 == 1 :
     print ("Processed", i, "files of ", len(files) )

np.save(outfile_feats, feats_array)
np.save(outfile_transcripts, transcripts_array)   
 





