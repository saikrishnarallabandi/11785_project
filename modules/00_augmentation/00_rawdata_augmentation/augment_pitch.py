import os, sys
import numpy as np
import soundfile as sf
import pyworld as pw

# Locations
data_dir = ''
target_dir = ''
files = sorted(os.listdir(data_dir))

for file in files:
    fname = file.split('.')[0]
    print fname
    
    # Extract features
    x, fs = sf.read(data_dir + '/' + file)
    f0, sp, ap = pw.wav2world(x, fs) 
    
    # Augment the f0
    f0 = f0 + 5
    
    # Synthesize
    y = pw.synthesize(f0, sp, ap, fs)

    # Save
    sf.write(target_dir + '/' + fname + '.wav')
    
