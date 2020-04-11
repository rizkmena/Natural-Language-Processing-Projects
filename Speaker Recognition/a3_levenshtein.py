import os
import numpy as np
import string
from scipy import stats

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """

    # Initializing R matrix
    n = len(r)
    m = len(h)

    R = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        R[i, 0] = i

    for j in range(1, m + 1):
        R[0, j] = j

    #Initializing backtrack matrix for keeping track of arrows
    backtrack_matrix = [[[0,0,0] for x in range(m+1)] for y in range(n+1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                R[i, j] = R[i - 1, j - 1]
                backtrack_matrix[i][j] = [backtrack_matrix[i - 1][j - 1][0],
                                          backtrack_matrix[i - 1][j - 1][1],
                                          backtrack_matrix[i - 1][j - 1][2]]
                continue

            #finding minimum number of errors
            errors = [R[i - 1, j - 1],   #substitution
                      R[i, j - 1],      #insertion
                      R[i - 1, j]]      #deletion

            R[i, j] = np.min(errors) + 1

            # updating backtrack matrix
            type_of_error = np.argmin(errors)

            if type_of_error == 0:
                update = [backtrack_matrix[i-1][j-1][0] + 1,
                          backtrack_matrix[i - 1][j - 1][1],
                          backtrack_matrix[i - 1][j - 1][2]]

            if type_of_error == 1:
                update = [backtrack_matrix[i][j - 1][0],
                          backtrack_matrix[i][j - 1][1] + 1,
                          backtrack_matrix[i][j - 1][2]]


            if type_of_error == 2:
                update = [backtrack_matrix[i - 1][j][0],
                          backtrack_matrix[i - 1][j][1],
                          backtrack_matrix[i - 1][j][2] + 1]

            backtrack_matrix[i][j] = update

    error_counts = backtrack_matrix[-1][-1]
    output = [R[n, m] / n] + error_counts
    return output

if __name__ == "__main__":

    punctuation = string.punctuation.replace('[', "").replace(']', "")

    f = open('asrDiscussion.txt', 'w')

    kaldi_wer = []
    google_wer = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            ref_path = os.path.join(dataDir, speaker, 'transcripts.txt')
            kaldi_path = os.path.join(dataDir, speaker,'transcripts.Kaldi.txt')
            google_path = os.path.join(dataDir, speaker,'transcripts.Google.txt')

            ref_lines = open(ref_path, 'r').read().splitlines()
            kaldi_lines = open(kaldi_path, 'r').read().splitlines()
            google_lines = open(google_path, 'r').read().splitlines()

            num_lines = len(ref_lines)
            for i in range(num_lines):

                ref_line = ref_lines[i].translate(str.maketrans('', '', punctuation)).lower().split(" ")
                kaldi_line = kaldi_lines[i].translate(str.maketrans('', '', punctuation)).lower().split(" ")
                google_line = google_lines[i].translate(str.maketrans('', '', punctuation)).lower().split(" ")

                kaldi_score = Levenshtein(ref_line, kaldi_line)
                google_score = Levenshtein(ref_line, google_line)

                print(f'{speaker} {"Kaldi"} {i} {kaldi_score[0]} S:{kaldi_score[1]}, I:{kaldi_score[2]}, D:{kaldi_score[3]}',file=f)
                print(f'{speaker} {"Google"} {i} {google_score[0]} S:{google_score[1]}, I:{google_score[2]}, D:{google_score[3]}',file=f)

                kaldi_wer.append(kaldi_score[0])
                google_wer.append(google_score[0])
            print('\n', file=f)

    print(f'Kaldi WER Average: {np.mean(kaldi_wer)} | Kaldi WER Standard Deviation: {np.std(kaldi_wer)}',file=f)
    print(f'Google WER Average: {np.mean(google_wer)} | Google WER Standard Deviation: {np.std(google_wer)}', file=f)
    print(f'T-test for Kaldi and Google WER Scores: {stats.ttest_ind(kaldi_wer, google_wer)}', file=f)
    f.close()
