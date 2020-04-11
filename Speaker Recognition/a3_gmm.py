from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = "/u/cs401/A3/data/"

seed = 20

np.random.seed(seed)
random.seed(seed)

class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """

        mth_row_sigma = self.Sigma[m,:]
        mth_row_mu_squared = np.power(self.mu[m,:],2)

        precomputed = np.sum((mth_row_mu_squared/(2*mth_row_sigma)))\
                      + (self._d/2)*np.log(2*np.pi)\
                      + ((1/2)*np.log(np.prod(mth_row_sigma)))

        return [precomputed]

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    precomputed_term = myTheta.precomputedForM(m)

    mth_row_sigma = myTheta.Sigma[m,:]
    mth_row_mu = myTheta.mu[m,:]
    log_bmx =  (-1)*(np.sum(
        ((1/2)*(np.power(x,2)/mth_row_sigma) - (mth_row_mu*x/mth_row_sigma)),axis = 1)) - precomputed_term

    return log_bmx



def stable_logsumexp(array_like, axis=-1):
    """Compute the stable logsumexp of `vector_like` along `axis`
    This `axis` is used primarily for vectorized calculations.
    """
    array = np.asarray(array_like)
    # keepdims should be True to allow for broadcasting
    m = np.max(array, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(array - m), axis=axis))

def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below

    """
    unnormalized = np.log(myTheta.omega) + log_Bs
    denom = stable_logsumexp(unnormalized, axis = 0)

    return unnormalized - denom

def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """

    terms = np.log(myTheta.omega) + log_Bs
    p_xt = stable_logsumexp(terms, axis=0)

    return np.sum(p_xt)

def computeIntermediateResults(M,X,theta):
    T = X.shape[0]
    log_Bs = np.empty((M,T))

    for m in range(M):
        log_Bs[m,:] = log_b_m_x(m,X,theta)

    log_p_Ms = log_p_m_x(log_Bs, theta)

    return log_Bs, log_p_Ms

def updateParameters(log_p_Ms, theta, X):
    T = X.shape[0]

    #recover p_M matrix
    p_Ms = np.exp(log_p_Ms)

    '''#perserve copy of sum of p_Ms with original dimensions for later operations
    sum_p_Ms_orig = np.sum(p_Ms,axis = 1)

    sum_p_Ms = np.expand_dims(sum_p_Ms_orig, axis=1)

    #update omega
    theta.omega = (sum_p_Ms)/T

    #stacking d copies of p_M matrix to vectorize update
    pm3d = np.repeat(p_Ms[:, :, np.newaxis], theta._d, axis=2)
    pm3d = np.swapaxes(pm3d,1,2).T

    # stacking M copies of x matrix to vectorize update
    x3d = np.repeat(X[:, :, np.newaxis], theta._M, axis=2)

    #Update mu
    unnormalized_mean = np.sum(pm3d*x3d,axis = 0)
    theta.mu = (unnormalized_mean/sum_p_Ms_orig).T

    #stacking M copies of x^2 to vectorize update of sigma
    x3dsquared = np.repeat((X**2)[:, :, np.newaxis], theta._M, axis=2)

    #udpate sigma
    theta.Sigma =  ((np.sum(pm3d*x3dsquared,axis = 0) / sum_p_Ms_orig) - ((theta.mu)**2).T).T + 0.000001'''

    p_Ms = np.exp(log_p_Ms)
    for m in range(M):
        # log of sum of p
        p_m = p_Ms[m]
        sum_p_m = np.sum(p_m)

        # update omega
        theta.omega[m] = sum_p_m / T

        # update mu
        theta.mu[m] = np.dot(p_m, X)/sum_p_m

        # update sigma
        theta.Sigma[m] = (np.dot(p_m, np.power(X,2))/sum_p_m) - (np.power(theta.mu[m],2)) + 0.000001

def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""

    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)

    #initialize omega
    w = np.random.rand(M,1)
    #ensure sum of terms in omega is 1
    myTheta.reset_omega(w/sum(w))

    #initialize mean with random sample from X
    sample = X[np.random.choice(X.shape[0], M, replace=False)]
    myTheta.reset_mu(sample)

    #Initialize sigma with each mth row equal to 1/m (as per slide 32)
    myTheta.reset_Sigma(np.ones((M, d))/np.expand_dims(np.arange(1,M+1),1))

    #training loop
    i = 0
    prev_L = -float('inf')
    improvement = float('inf')
    while i <= maxIter and improvement >= epsilon:
        log_Bs, log_p_Ms = computeIntermediateResults(M,X,myTheta)
        L = logLik(log_Bs, myTheta)
        updateParameters(log_p_Ms, myTheta, X)
        improvement = L - prev_L
        prev_L = L
        i += 1

    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    L = []


    best_pred = -1000000000000000000000000000000
    i = 0
    preds = []
    for model in models:
        M = model.Sigma.shape[0]
        log_Bs, log_p_Ms = computeIntermediateResults(M, mfcc, model)
        res = logLik(log_Bs, model)
        preds.append(res)
        L.append([model.name, res])
        if res > best_pred:
            best_pred = res
            bestModel = i
        i += 1

    sorted_L = sorted(L,key = lambda x:x[1],reverse = True)


    print(models[correctID].name)
    for i in range(k):
        print(sorted_L[i][0], sorted_L[i][1])
    print('\n')


    # with open("./gmmLiks.txt", "w") as f:
    #     f.write(f"{model[correctID]}\n")
    #     for i in range(k):
    #         print(correctID, sorted_L[i][0],sorted_L[i][1])
    #         f.write(f"{sorted_L[i][0]} {sorted_L[i][1]}\n")
    #         #f.write('{} {}'.format(sorted_L[i][0],sorted_L[i][1]))
    #     f.write("\n")

    #bestModel = msorted_L[0][0]
    #print(bestModel, correctID)

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []

    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            #print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    #print(accuracy)
