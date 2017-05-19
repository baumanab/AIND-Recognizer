import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
		   
    ## General Strategy:
    Essentially we want to find the best guess for each sequence in the test set.
	
	- Get test sequenes and initialize probabilities and guesses
	    + For each sequence and word index in the test sequences
		    ++ get X and lengths for each word index 
			++ Apply X and lengths for each model in the trained models dict
			   This will give the log likelihood (probabilty) of that sequence
			   for each word in the model dictionary. We pass the word and the 
               probabitly to probabilties, we pass the highest probabilty word 
               for each sequence to guesses.
			   
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # get all sequences in the test set
    all_sequences= test_set.get_all_sequences()

    # iterate over sequences get X and lengths for each key
    for index, sequence in all_sequences.items():
        X, lengths= test_set.get_item_Xlengths(index)
        guess_probs= dict() # container to hold the word:probabilities for each sequnence for each model

      # iterate over each word:model pair where the model is the trained model for that word
        for word, word_model in models.items():
            try:
                prob= word_model.score(X, lengths)
                guess_probs[word]= prob
            except:
                guess_probs[word]= float('-inf') # lowest possible score

        probabilities.append(guess_probs) # add dictonary to list
        # get return key with highest score as best guess
        # thanks stack overflow! http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        best_guess= max(guess_probs, key=guess_probs.get)
        guesses.append(best_guess)

    return probabilities, guesses
