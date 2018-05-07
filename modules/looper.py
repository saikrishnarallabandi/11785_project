import os

file = 'train_transcripts.txt'

sentences = []
f = open(file)
for line in f:
   line = line.split('\n')[0]
   sentences.append(line)

length = len(sentences)
print length
counts = range(0,length, 1000)
print counts
for count in counts:
    selected = sentences[0:count]
    words = []
    for sentence in selected:
        wds = sentence.split()
        for word in wds:
            words.append(word)
    print "Count: ", count,  "Sentences: ", len(selected), " Words, " , len(words), "unique words: ", len(set(words))

