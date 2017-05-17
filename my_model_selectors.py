import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
		
		- great forum resources to inform development
        - https://discussions.udacity.com/t/verifing-bic-calculation/246165/2
        - https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/14
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
		
		


        # initialize variables with score being the largest possible number
        best_score= float("inf")
        best_model= None
        
        # get other attributes we need for BIC calculation
        num_features= len(self.X[0])
        N= np.sum(self.lengths)
        logN= np.log(N)
      
        for component in range(self.min_n_components, self.max_n_components + 1):
            try:
                
                # get model and log likelihood
                model= self.base_model(component) #GaussianHMM
                logL= model.score(self.X, self.lengths)

                # calculate parameters
                p= (component**2) + 2 * num_features * component - 1

                # calculate BIC
                score= -2 * logL + p * logN

                # update score and model that generated score
                if score < best_score:
                    best_score= score
                    best_model= model

            except:
                print("failure on {} @ {}".format(self.this_word, component))

        return best_model
        
        
        
        
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        # initialize essential objects
        split_method= KFold()

        # outer loop iterating over components
        for components in range(self.min_n_components, self.max_n_components + 1):

            # grab essential objects
            word_sequences= self.sequences
            num_seqences= len(word_sequences) # count of word sequences
            n_splits= min(10, num_seqences) # 10 splits max, num_seqences splits min
			
			mean_scores= list() # create array to store mean cv scores from inner loop

            try:
                splitter= split_method(n_splits)
            except:
                return None

            # inner loop where CV takes place
            try:			    
			    for cv_train_idx, cv_test_idx in splitter.split(word_sequences):
				    local_scores= list() # create array to store cv scores
				    # use indices to get train and test set array and length
				    cv_train_x, cv_train_length= combine_sequences(cv_train_idx, word_sequences)
					cv_test_x, cv_test_length= combine_sequences(cv_test_idx, word_sequences)
					# build a model using the cv traning data
					cv_model= GaussianHMM(n_components=components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(cv_train_x, cv_train_length)
					# get the model score (log likelihood) for the test fold
					cv_score= cv_model.score(cv_test_x, cv_train_x)
					local_scores.append(cv_score)
					

            except:
                pass

