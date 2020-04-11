import numpy as np
import argparse
import json
import pandas as pd
import re
from tqdm import tqdm

#Word and tag lists
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
COORD_CONJ = {
    'CC'}
PAST_TENSE_VERB = {
    'VBD'}
PUNCT = {
    'NFP'}
COMMON_NOUN = {
    'NN', 'NNS'}
PROPER_NOUN = {
    'NNP', 'NNPS'}
ADVERB = {
    'RB', 'RBR', 'RBS'}
WH_WORDS = {
    'WDT', 'WP', 'WP$', 'WRB'}


#Read the Bristol, Gilhooly, and Logie norm file and extract the values
#for reduced computation during future feature processing steps
BGL_NORMS = pd.read_csv("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv")
BGL_NORMS = BGL_NORMS[['WORD','AoA (100-700)', 'IMG', 'FAM']].values

#Read the Warringer norm file and extract the values
#for reduced computation during future feature processing steps
W_NORMS = pd.read_csv("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv")
W_NORMS = W_NORMS[['Word','V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']].values

#Load the LIWC/Receptiviti features for each of the classes
ALT_FEATS = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")
CTR_FEATS = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
LFT_FEATS = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
RGT_FEATS = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")

#Load the comment IDs for each of the classes
ALT_IDS = np.loadtxt("/u/cs401/A1/feats/Alt_IDs.txt", dtype = str)
CTR_IDS = np.loadtxt("/u/cs401/A1/feats/Center_IDs.txt", dtype = str)
LFT_IDS= np.loadtxt("/u/cs401/A1/feats/Left_IDs.txt", dtype = str)
RGT_IDS = np.loadtxt("/u/cs401/A1/feats/Right_IDs.txt", dtype = str)



def sentence_splitter(body):
    ''' This helper function processes each "token/TAG" string in the comment as a list
    of ["token","tag"].

    Parameters:
        body : string, the body of a comment (after preprocessing)

    Returns:
        split_comment : list of each [token/tag] in the comment
    '''    

    #splits comment on a space character to
    #identify tokens
    tokens = body.strip().split(' ')
   
    
    split_comment = []
    
    #iterate through each token in body
    #and append["token","tag"] to the processed body list
    for t in tokens:
        split = t.split('/')
        split_comment.append(split)
    
    return split_comment
    


def future_tense_counter(comment,split_comment):
    ''' This helper function counts the amount of future tense verbs.
    
    Parameters:
        comment : string, the body of a comment (after preprocessing)
        split_comment : list of each [token/tag] in the comment

    Returns:
        total_count : integer count of future tense verbs
    '''    
    
    #list of words that appear before verbs
    word_list = ["'ll", 'will', 'gonna']
    
    #Find all occurences in the comment of the words in word_list.
    #And then count the amount of occurences.
    matches = re.findall(r'\b' + r'/|\b'.join(word_list) + r'/', comment, flags=re.IGNORECASE)
    count1 = len(matches)
    
    #Use the split_comment parameter to count sequences of 
    #"{go, going} to" followed by a verb
    count2 = 0
    for t in range(len(split_comment)-2):
        if (split_comment[t][0] in ['go', 'going'] and split_comment[t+1][0] == 'to' and split_comment[t+2][1] == 'VB'):
            count2 += 1
    
    #return a sum of both counts for a total count of verbs in comment
    total_count = count1 + count2
    return total_count
    

def avg_sentence_length(body):
    ''' This helper function computes the average sentence length
    
    Parameters:
        body : string, the body of a comment (after preprocessing)

    Returns:
        avg : the average sentence length of the given comment
    '''   
    
    #Count of tokens in body though counting spaces
    token_count = len(re.findall(r"\s", body, flags=re.IGNORECASE))
    
    #Count of sentences in body though counting newline characters
    sentences = len(re.findall(r"\n", body, flags=re.IGNORECASE))
    
    #in the case of zero sentences, return 0 to avoid a division error
    if sentences == 0:
        return 0
        
    #return average 
    avg = token_count/sentences 
    return avg
            
        
def avg_token_length(body):
    ''' This helper function computes the average token length
    
    Parameters:
        body : string, the body of a comment (after preprocessing)

    Returns:
        avg : the average token length of the given comment
    '''   
    
    #substitutes all space characters with tabs this
    #method was used for identifying the first token in the
    #comment during the next step of extracting token lengths
    body = re.sub(r'\s', r'\t', body)
    
    #searches for word characters encapsulated by a tab and slash
    #note the addition of '\t' to the body to ensure the first token
    #is also counted
    split_token_tag = re.findall(r'\s[\S]*\w+[\S]*/', '\t'+body)
    
    
    token_length_sum = 0
    for t in split_token_tag:
        #count the amount of characters in the token and subtract 2
        #(subtracts the inclusion of the tab and the slash after each token)
        token_length_sum += len(t) - 2
    
    #if there are no tokens, return o to avoid division error
    if len(split_token_tag) == 0:
        return 0
    
    avg = token_length_sum/len(split_token_tag)
    return avg
    

def sentence_counter(body):
    ''' This helper function computes the count of sentences in a given comment
    
    Parameters:
        body : string, the body of a comment (after preprocessing)

    Returns:
        count : integer count of number of sentences in the given comment
    '''   
    
    #count and return the number of newline chracters in the comment
    sentences = body.strip().split('\n')
    count = len(sentences)
    return count

  
def norm_list(body):
    ''' This helper function finds the AoA, IMG, FAM,
     V.Mean.Sum, A.Mean.Sum, and D.Mean.Sum values of tokens present
    in the body.
    
    Parameters:
        body : list of all tokens in the comment where each token is a list of [token,tag]

    Returns:
        [AoA, IMG, FAM, v_mean_sum, a_mean_sum, d_mean_sum] : list of 6 lists (for each of
        the six features where each inner list contains all of the corresponding feature's
        values that were present in the given comment.
    '''

    AoA = []
    IMG = []
    FAM = []
    v_mean_sum = []
    a_mean_sum = []
    d_mean_sum = []

    #Iterate through each token in the comment
    for t in body:

       #Extact the word from the [token/tag] list and lowercase the word
        word = t[0].lower() 

        #list of all words in the BGL and Warringer norm files
        bgl_words = BGL_NORMS[:,0]
        w_words = W_NORMS[:,0]

        #if the selected word occurs in the BGL list, extract the row of the word
        #in the BGL list and then append the AoA, IMG, and FAM values to their corresponding lists
        if word in bgl_words:

            i,j= np.where(BGL_NORMS ==word)
            row = BGL_NORMS[i[0],:]
            AoA.append(row[1])
            IMG.append(row[2])
            FAM.append(row[3])

        #if the selected word occurs in the Warringer list, extract the row of the word
        #in the Warringer list and then append the V.Mean.Sum, A.Mean.Sum, and D.Mean.Sum values
        #to their corresponding lists
        if word in w_words:
            
            i,j= np.where(W_NORMS == word)
            row = W_NORMS[i[0],:]
            v_mean_sum.append(row[1])
            a_mean_sum.append(row[2])
            d_mean_sum.append(row[3])
        
    return [AoA, IMG, FAM, v_mean_sum, a_mean_sum, d_mean_sum]
    
def norm_feats(body):
    ''' This helper function finds the means and standard deviations for each of the BGL and Norm feautures

    Parameters:
        body : list of all tokens in the comment where each token is a list of [token,tag]

    Returns:
        feats : list of features #18-#29 (in order of the assignment document)
    '''

    #Exract all of the present norm features using the norm_list function
    norm_lists = norm_list(body)

    #Compute and return the BGL feature means, BGL feature standard deviations,
    #Warringer feature means, and Warringer feature standard deviations
    feats = [np.mean(norm_lists[0]),np.mean(norm_lists[1]),
            np.mean(norm_lists[2]),np.std(norm_lists[0]),
            np.std(norm_lists[1]), np.std(norm_lists[2]),
            np.mean(norm_lists[3]),np.mean(norm_lists[4]),
            np.mean(norm_lists[5]),np.std(norm_lists[3]),
            np.std(norm_lists[4]), np.std(norm_lists[5])]
    return feats

    
def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    

    word_feat_list = [FIRST_PERSON_PRONOUNS, SECOND_PERSON_PRONOUNS,
    THIRD_PERSON_PRONOUNS]
    
    tag_feat_list = [PUNCT, COMMON_NOUN, PROPER_NOUN, ADVERB, WH_WORDS]

    feats = []
    split_comment = sentence_splitter(comment)

    
    #Extract feature #1
    capital_matches = re.findall(r'[A-Z][A-Z][A-Z]+/',comment)
    feats.append(len(capital_matches))    
    
    #Extract features #2 - #4
    for feat in word_feat_list:
        matches = re.findall(r'\b' + r'/|\b'.join(feat) + r'/', comment, flags=re.IGNORECASE)
        feats.append(len(matches))
    
    #Extract feature #5
    coord_conj_matches = re.findall(r'/' + r'\b|/'.join(COORD_CONJ)+r'\b', comment, flags=re.IGNORECASE)
    feats.append(len(coord_conj_matches))
    
    #Extract feature #6
    past_tense_matches = re.findall(r'/' + r'\b|/'.join(PAST_TENSE_VERB)+r'\b', comment, flags=re.IGNORECASE)
    feats.append(len(past_tense_matches))
    
    #Extract feature #7
    feats.append(future_tense_counter(comment, split_comment))
    
    #Extract feature #8
    comma_matches = re.findall(r',/', comment, flags=re.IGNORECASE)
    feats.append(len(comma_matches))
    

    #Extract features #9 - #13
    for feat in tag_feat_list:
        matches = re.findall(r'/' + r'\b|/'.join(feat)+r'\b', comment)
        feats.append(len(matches))
        
    #Extract feature #14
    slang_matches = re.findall(r'\b' + r'/|\b'.join(SLANG) + r'/', comment, flags=re.IGNORECASE)
    feats.append(len(slang_matches))

    #Extract feature #15
    feats.append(avg_sentence_length(comment))
    
    #Extract feature #16
    feats.append(avg_token_length(comment))
    
    #Extract feature #17
    feats.append(sentence_counter(comment))
    
    #Extract features #18 - #29
    feats = feats + norm_feats(split_comment)
    
    #Padding the feature array with 144 zeros for the LIWC/Receptiviti features to be extracted using extract2
    feats = np.asarray(feats)
    feats = np.concatenate((feats, np.zeros(144)))
    
    return feats
    
    
def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''

    txt_classes = {'Alt': ALT_IDS, 'Center': CTR_IDS , 'Left': LFT_IDS, 'Right': RGT_IDS}
    feat_classes = {'Alt': ALT_FEATS, 'Center': CTR_FEATS , 'Left':LFT_FEATS, 'Right': RGT_FEATS}

    #Find the index of the comment's ID from the ID list of the corresponding class
    idx = np.where(txt_classes[comment_class] == comment_id)

    #Extract the row from the feature list of the class' corresponding feature list
    liwc_feats = feat_classes[comment_class][idx]
    feats[29:] = liwc_feats
    
    return feats

def main(args):
    #Load the processed json file
    data = json.load(open(args.input))

    #Initialize feature matrix
    feats = np.zeros((len(data), 173+1))

    class_codes = {'Alt': 3, 'Center':1 , 'Left':0, 'Right':2}

    #for each datum, extract features
    idx = 0
    for j in tqdm(data):
        
        datum_id = j['id']
        datum_body = j['body']
        datum_class = j['cat']
        
        feats1 = extract1(datum_body)
        feats2 = extract2(feats1, datum_class, datum_id)

        #encode the datum label using the class_codes dictionary and input into data matrix
        feats[idx][:173] = feats2
        feats[idx][173] = class_codes[datum_class]
        
        idx += 1

    #eleminate all nan values in data matrix
    feats = np.nan_to_num(feats)  
       
    #save data matrix as npz file
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

