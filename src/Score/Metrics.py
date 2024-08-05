import numpy as np


class Metrics():
    
    @staticmethod
    def WER(reference, hypothesis):

        ref_words = reference.split()

        hyp_words = hypothesis.split()
        
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

        for i in range(len(ref_words) + 1):
            d[i, 0] = i

        for j in range(len(hyp_words) + 1):
            d[0, j] = j

        for i in range(1, len(ref_words) + 1):

            for j in range(1, len(hyp_words) + 1):

                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:

                    substitution = d[i - 1, j - 1] + 1
                    insertion = d[i, j - 1] + 1
                    deletion = d[i - 1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)

        wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
        
        return wer