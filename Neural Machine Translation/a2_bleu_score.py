# Copyright 2020 University of Toronto, all rights reserved

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    
    '''
    
    ngrams = []
    
    for i in range(len(seq) - n + 1):
        ngram = []
    	
        for j in range(n):
    	    ngram.append(seq[i+j])
    		
        ngrams.append(ngram)
    	    
    
    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''

    if len(candidate) == 0:
        return 0
    
    
    ref_ngrams = grouper(reference, n)
    cand_ngrams = grouper(candidate, n)
    
    C = 0
    
    for ngram in cand_ngrams:
        if ngram in ref_ngrams:
            C += 1
    
    
    p_n = C/len(cand_ngrams)

    return p_n
    
    

def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    if len(candidate) == 0:
        return 0
    
    brev = len(reference)/len(candidate)
    if brev < 1:
        BP = 1
        
    else:
        BP = exp(1-brev)
        
    return BP


def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    precision_product = 1
    
    
    for i in range(1,n+1):
        p = n_gram_precision(reference, hypothesis, i)
        precision_product = precision_product*p
   
    
    BP = brevity_penalty(reference, hypothesis)
    
    bleu = BP*(precision_product**(1/n))
    
    return bleu
 
        

    
    
