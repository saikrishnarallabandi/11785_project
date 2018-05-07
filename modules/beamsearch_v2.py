        sequences = [[list(), 1.0]]
        #print "Shape I got is ", probs.shape
        for row in probs:
            #print "Row I got is ", row
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * -log(row[j])]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:13]
        #print "I am returning", sequences
        seq, score = sequences[0]
