import sys
import argparse
import os
import json
import re
import spacy
import pprint
import html
from tqdm import tqdm


#initialation of Spacy document
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 5)):

    modComm = comment
    
    # replace newlines with spaces
    if 1 in steps:  
        modComm = re.sub(r"\n{1,}", " ", modComm)
        
    # unescape html (i.e. remove HTML character codes)
    if 2 in steps:  
        modComm = html.unescape(modComm)
    
    # remove URLs
    if 3 in steps:  
        modComm = re.sub(r"(http|www)\S+", "", modComm)
        
    # remove duplicate spaces
    if 4 in steps:  
        modComm = re.sub(' +', ' ',modComm)

    #get Spacy document for modComm
    utt = nlp(modComm)

    #initialization of empty string for storing modified
    #comment string
    comment = ""
    
    #iterate through every sentence in Spacy document for
    #replacing each token with its lemma, appending the
    #corresponding tag for each token, and adding a newline
    #character between each sentence
    for sent in utt.sents:
        
        #initialization of empty string for storing modified
        #sentence string
        temp_sent = ""
    	
        
        for token in sent:
              
                #Replace each token with lemma unless it starts
                #with "-". Then append the POS tag to each token
        	if token.lemma_[0] == "-":
        		t = token.text + "/" + token.tag_
        		
        	else:
        		t = token.lemma_ + "/" + token.tag_
        		
        	#Append the modified token to the modified sentence	
        	temp_sent += (t + " ")
        
        #Append the modified sentence to the the em	
        comment += (temp_sent + "\n")
        
    modComm = comment

   
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)
            
            
            #Load the unprocessed json file
            data = json.load(open(fullFile))
            
            
            #Subsampling of 10000 data samples from each category using student ID
            data = data[(args.ID[0]%len(data)):((args.ID[0]%len(data)) + args.max)]
           
            
            c = 0
            
            for line in tqdm(data):
            
                #read the selected line
            	j = json.loads(line)
            	
            	
            	line_id = j['id']
            	line_body = j['body']
            	
            	
            	#remove all fields and input desired fields
            	j.clear()
            	j['id'] = line_id
            	#replace the 'body' field with the processed text
            	j['body'] = preproc1(line_body)
            	
            	#add a category field for each selected line.
            	#this field stores the value of file.
            	j['cat'] = file

            	#append the result to 'allOutput'
            	allOutput.append(j)
            	
            	c+=1
            	


            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput,indent=4, sort_keys=True))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
