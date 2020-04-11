import argparse
import os
import sys
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
import numpy as np

def accuracy(C):
    ''' This function computes the accuracy of a model using its
    corresponding confusion matrix.

    Parameters:
        C : confusion matrix

    Returns:
        acc : accuracy of model as float
    '''

    #Sum of all correctly classified samples (diagonals of C)
    num = np.trace(C)

    #Sum of all samples
    denom = np.sum(C)

    acc = num/denom
    return acc


def recall(C):
    ''' This function computes the recall of a model using its
    corresponding confusion matrix.

    Parameters:
        C : confusion matrix

    Returns:
        r : list containing the recall of a model for each of the classes
    '''

    #initialize recall list
    r = []

    #Computes row sums
    row_sums = np.sum(C, axis = 0)

    #computes the recall for each of the classes and appends the result to r
    for k in range(4):
        r.append(C[k][k]/row_sums[k])
    
    return r

def precision(C):
    ''' This function computes the precision of a model using its
    corresponding confusion matrix.

    Parameters:
        C : confusion matrix

    Returns:
        p : list containing the precision of a model for each of the classes
    '''

    #initilaize precision list
    p =  []

    #computes column sums
    col_sums = np.sum(C, axis = 1)

    #computes the precision for each of the classes and appends the result to p
    for k in range(4):
        p.append(C[k][k]/col_sums[k])
        
    return p
    
    
def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''

    #initialize each of the 5 classifiers
    clf1 = SGDClassifier()
    clf2 = GaussianNB()
    clf3 = RandomForestClassifier(n_estimators = 10, max_depth = 5)
    clf4 = MLPClassifier(alpha = 0.05)
    clf5 = AdaBoostClassifier()

    #store the classifiers in a list with their corresponding names (for printing)
    classifiers = [[clf1,"SGDClassifier"], [clf2,"GaussianNB"],
                   [clf3,"RandomForestClassifier"],[clf4,"MLPClassifier"],
                   [clf5,"AdaBoostClassifier"]]
    

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        iBest = 0
        cur = 0
        max_acc = 0


        #iterate through each classifier
        for clf in classifiers:

            #train the classifier
            clf[0].fit(X_train, y_train)

            #compute predictions on the test set
            pred = clf[0].predict(X_test)

            #produce confusion matrix of model predictions on test set
            C = confusion_matrix(y_test,pred)

            #Write results to file
            outf.write(f'Results for {clf[1]}:\n')
            outf.write(f'\tAccuracy: {round(accuracy(C),4)}\n')
            outf.write(f'\tRecall: {np.around(recall(C),decimals = 4)}\n')
            outf.write(f'\tPrecision: {np.around(precision(C),decimals = 4)}\n')
            outf.write(f'\tConfusion Matrix: \n{C}\n\n')

            #Identify the index of the model with the highest accuracy
            if round(accuracy(C),4) > max_acc:
                iBest = cur
                max_acc = round(accuracy(C),4)
            cur +=1


    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
  
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''


    #list of all training sizes to train the classifiers on
    train_sizes= [1000,5000,10000,15000,20000]

    # initialize each of the 5 classifiers
    clf1 = SGDClassifier()
    clf2 = GaussianNB()
    clf3 = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf4 = MLPClassifier(alpha=0.05)
    clf5 = AdaBoostClassifier()

    # store the classifiers in a list with their corresponding names
    classifiers = [[clf1, "SGDClassifier"], [clf2, "GaussianNB"],
                   [clf3, "RandomForestClassifier"], [clf4, "MLPClassifier"],
                   [clf5, "AdaBoostClassifier"]]

    #store the best classifier in the best variable
    best = classifiers[iBest]

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:

        #Iterate through each training size and train the best model using the selected training sieze
        for train_size in train_sizes:

            #use train_test_split to randomly sample the selected amount of training samples
            X_scrap, X_sample, y_scrap, y_sample = train_test_split(X_train, y_train, test_size = int(train_size))

            #Store the training set of size 1000
            if train_size == 1000:
                X_1k = X_sample
                y_1k = y_sample

            #train the classifier
            best[0].fit(X_sample, y_sample)

            #compute predictions on the test set
            pred = best[0].predict(X_test)

            #produce confusion matrix of the test set results
            C = confusion_matrix(y_test,pred)

            #write results to file
            outf.write(f'{train_size}: {round(accuracy(C),4)}\n')
        outf.write(f"\n\nIt can be observed that as the amount of training samples increase,\n the test accuracy of the classifier also increases.\n \nThis trend is expected since I hypothesize that an increase in training\nsamples increases the model's ability to generalize to unseen comments \nsince it reduces the liklihood of the classifier learning spurious \ncorrelations in the features of the training comments. \nAlso, with an increase in the amount of training samples, the classifier places less of a bias on each of the individual samples, enabling it to better generalize.")

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    # initialize each of the 5 classifiers
    clf1 = SGDClassifier()
    clf2 = GaussianNB()
    clf3 = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf4 = MLPClassifier(alpha=0.05)
    clf5 = AdaBoostClassifier()

    # store the classifiers in a list with their corresponding names
    classifiers = [[clf1, "SGDClassifier"], [clf2, "GaussianNB"],
                   [clf3, "RandomForestClassifier"], [clf4, "MLPClassifier"],
                   [clf5, "AdaBoostClassifier"]]

    # store the best classifier in the best variable
    best = classifiers[i]

    k_list = [5,50]

    #initiailize pvalue list
    pval_list = []

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.


        #3.3.1
        #iterate through each amount of k values
        for k in k_list:
        
            #select k best features
            selector = SelectKBest(f_classif, k = k)

            #train the model using the k best features
            selector.fit(X_train,y_train)

            #extract the p_value
            p_values = selector.pvalues_
            pval_list.append(p_values)
            outf.write(f'{k} p-values: {[round(pval, 4) for pval in p_values]}\n')

        #3.3.2.
        top_5_32k = np.argsort(pval_list[0])[:5]
        X_train_top_5 = X_train[:,top_5_32k]
        
        #get 5 best parameters for the 1K training set
        selector_1k = SelectKBest(f_classif, k=5)
        selector_1k.fit(X_1k, y_1k)
        
        #Extract the k=5 best features for the 1K training set
        p_values_1k = selector_1k.pvalues_
        top_5_1k = np.argsort(p_values_1k)[:5]
        X_1k_top_5 = X_1k[:,top_5_1k]

        #Produce the k=5 best featurte test sets for the 32k and 1k training sets
        X_test_32k_top_5 = X_test[:,top_5_32k]
        X_test_1k_top_5 = X_test[:, top_5_1k]


        #training the best classifier on the 32k and the 1k training sets
        
        #32k
        clf_32k_top_5 = best[0]
        clf_32k_top_5.fit(X_train_top_5,y_train)
        #1k
        clf_1k_top_5 = best[0]
        clf_1k_top_5.fit(X_1k_top_5,y_1k)
       
        #Compute predictions for both classifiers
        pred_32k = clf_32k_top_5.predict(X_test_32k_top_5)
        pred_1k = clf_1k_top_5.predict(X_test_1k_top_5)
        
        #produce confusion matrices for the results of both classifers on their
        #corresponding test sets
        C_32k = confusion_matrix(y_test, pred_32k)
        C_1k = confusion_matrix(y_test, pred_1k)

        outf.write(f'Accuracy for 1k: {round(accuracy(C_32k),4)}\n')
        outf.write(f'Accuracy for full dataset: {round(accuracy(C_1k),4)}\n')

        #3.3.2.
        feature_intersection = np.intersect1d(top_5_32k,top_5_1k)     
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        
        #3.3.4.
        outf.write(f'Top-5 at higher: {top_5_32k}\n')
        outf.write(f"\na) Names of features in the intersection:\n(Note each feature is incremented by 1 since the indexing starts at 0)\n   #21 - Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms\n   #150 - receptiviti_intellectual \n   #164 - receptiviti_self_conscious\n\nb)Since a lower p-value means that the result is less likely to be due to pure chance, I hypothesize that with more training data, the p-values are generally lower since an increase in the amount of training data reduces the liklihood of spurious correlations.\n\nc) Top 5 features for the 32K training case:\n(Note each feature is incremented by 1 since the indexing starts at 0)\n\n#12 - Number of adverbs\nThere may be a relationship between the reasoning style of comments and political leaning. This could be captured with the number of adverbs since it would capture words like 'however' which may be associated with the explanation/writing/reasoning style that is more common amongst commenters of a particular political leaning\n\n#21 - Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms\nAssuming there exists some relationship between the age of commenters and political leaning,\nthis feature may be able to differentiate between classes through providing a rough proxy of the\nage of the commenter since it gives a measure of the distribution of the age of acquisiton\nof the words in the text.\n\n#22 - Standard deviation of IMG from Bristol, Gilhooly, and Logie norms\nThis feature gives a rough measure for the distribution of imagery in a comment. This can\npossibly distinguish between classes since there may be a correlation between political\nleaning and the imagery in the writing style of the commenters. For example,\ncommenters of some political class may have a vocabulary of predominantly high or low imagery or\na combination of both, this would be captured with this feature.\n\n#150 - receptiviti_intellectual\nThere may also exist a relationship between political leaning and the writing style in a\ncomment from the sense of how appeals in the text are made. Particularly, there may exist a\ndisproportionately larger amount of commenters from a particular poltical leaning who appeal to\nintellect or use a more intellectual vocabulary/writing style in the text. As such, this measure\nwould be able to differentiate between classes using this feature.\n\n#164 - receptiviti_self_conscious\nSimilarly to feature #150, there may also exist a relationship between the measure of\n'self_concsiousness' and the poltical leaning of a comment. For example, one poltical leaning\nclass may be predominantly composed of comments with very little self consiousness whereas\nanother class may have primarly comments with more self-consiousness. The classifier would then\nbe able to use this feature to distinguish between classes.")




def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''

    # initialize each of the 5 classifiers
    clf1 = SGDClassifier()
    clf2 = GaussianNB()
    clf3 = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf4 = MLPClassifier(alpha=0.05)
    clf5 = AdaBoostClassifier()

    # store the classifiers in a list with their corresponding names
    classifiers = [[clf1, "SGDClassifier"], [clf2, "GaussianNB"],
                   [clf3, "RandomForestClassifier"], [clf4, "MLPClassifier"],
                   [clf5, "AdaBoostClassifier"]]

    #store the best classifier in the best variable
    best = classifiers[i]

    #combine the training and test sets for k-fold splitting
    X = np.vstack((X_train,X_test))
    y = np.hstack((y_train,y_test))

    #split the entire dataset into 5 folds
    k5fold = KFold(n_splits = 5, shuffle = True)
    k5fold.get_n_splits(X)
    
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        
        #initialize accuracy matrix
        acc_matrix = []
        
        #iterate through each fold
        for train_index, test_index in k5fold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            kfold_accuracies = []
            
            #iterate through each classifier
            for clf in classifiers:
                #train the classifier
                clf[0].fit(X_train, y_train)
                
                #compute predictions on the test set
                pred = clf[0].predict(X_test)
                
                #produce the confusion matrix on the results of the test set
                C = confusion_matrix(y_test,pred)
                
                #append the accraucy to the kfold_accuracies list
                kfold_accuracies.append(accuracy(C))
                
            #append the kfold accuracies to the accuracy matrix   
            acc_matrix.append(kfold_accuracies)
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        
        #convert the matrix to numpy matrix
        acc_matrix = np.asarray(acc_matrix)

        #comparison
        p_values = []
        for col in range(0,5):
            if col == i:
                continue
                
            #compute the pvalues
            S = ttest_rel(acc_matrix[:,col],acc_matrix[:,i])
            p_values.append(S.pvalue)
            
     
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    #load and extract the data set
    npz = np.load(args.input)
    data = npz['arr_0']
    
    #seperate the features and labels in the data matrix
    feats = data[:, :-1]
    labels = data[:, -1]

    #Create the random
    np.random.seed(401)
    
    #Run experiments 3.1. to 3.4.
    X_train, X_test, y_train, y_test = train_test_split(feats, labels, test_size=0.2)
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)

