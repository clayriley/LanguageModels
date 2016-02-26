/**
 * 
 */
package cs114.langmodel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import cs114.util.Counter;
import cs114.util.Counters;
import cs114.util.Pair;

/**
 * @author clay riley
 * 
 */
public class BigramLaplace extends LanguageModel {

	private Counter<Pair<String, String>> probCounter = new Counter<Pair<String, String>>();
    private Set<String> vocabulary;
	private Counter<String> tokens = new Counter<String>(); // counter for unigrams
    // private Counter<String> lps; // or some other dictionary structure
	private double totalTokens;
	private double totalBigrams; // TODO
	
	/**
	 * 
	 */
	public BigramLaplace() {
		// Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see cs114.langmodel.LanguageModel#train(java.util.Collection)
	 */
	@Override
	public void train(Collection<List<String>> trainingSentences) {
		// create a collection of strings representing a bigram
		Counter<Pair<String, String>> bigramCounter = new Counter<Pair<String, String>>(); // TODO swap this out for the probCounter
		vocabulary = new TreeSet<String>();
		for (List<String> sent : trainingSentences) { // for each sentence in training data...
			for (int i = 0; i < sent.size(); i++){ // loop through the sentence
				if (i==0){
					Pair<String, String> bigram = new Pair<String, String>(START,sent.get(i)); // first bigram is the START plus the first word
					bigramCounter.incrementCount(bigram, 1.0);
					tokens.incrementCount(START, 1.0);
					tokens.incrementCount(sent.get(i), 1.0);
				}
				else { 
					Pair<String, String> bigram;
					if (i==sent.size()-1){ // if the last word is reached,
						bigram = new Pair<String, String>(sent.get(i),STOP); // last bigram is the final word plus the STOP.
						// update the count of tokens in this training set with STOP
						tokens.incrementCount(STOP, 1.0);
						tokens.incrementCount(sent.get(i), 1.0);
					}
					else {
						bigram = new Pair<String, String>(sent.get(i),sent.get(i+1)); // all other bigrams are this word plus the next one
						// update the count of tokens in this training set with the current word
						tokens.incrementCount(sent.get(i), 1.0);
					}
					bigramCounter.incrementCount(bigram, 1.0); // increment the bigram counter
				}//TODO get rid of the unk stuff above and below here or at least figure out what it's doing
			}
		}
		/*
		 * 
		// we also need to include the UNKNOWN pairings for each word.
		for (String w : vocabulary){
			bigramCounter.incrementCount(w+" "+UNK, 1.0); // 1.0 == Laplace smoothing
			bigramCounter.incrementCount(UNK+" "+w, 1.0); // 1.0 == Laplace smoothing
		}
		// don't forget the other unknown cases!
		bigramCounter.incrementCount(UNK+" "+UNK, 1.0); // 1.0 == Laplace smoothing
		bigramCounter.incrementCount(UNK+" "+STOP, 1.0); // 1.0 == Laplace smoothing
		bigramCounter.incrementCount(START+" "+UNK, 1.0); // 1.0 == Laplace smoothing
		// add UNK and STOP to the vocabulary
		 * 
		 */
		tokens.incrementCount(UNK, 1.0);
		totalTokens = tokens.totalCount();
		
		vocabulary.addAll(tokens.keySet());
		vocabulary = Collections.unmodifiableSet(vocabulary);
		
		// generate the probability distribution by normalizing the counts to between 0 and 1
		probCounter = bigramCounter;
		totalBigrams = bigramCounter.totalCount();
		
		
	}

	/* (non-Javadoc)
	 * @see cs114.langmodel.LanguageModel#getWordProbability(java.util.List, int)
	 */
	@Override
	public double getWordProbability(List<String> sentence, int index) {		
		// assign context and word, looking behind
		String context;
		String w = sentence.get(index);
		if (index == 0){
			context = START;
		}
		else {
			context = sentence.get(index-1);
		}
		/*
		 * P_Laplace(w_n|w_n−1) = (C(w_n−1, w_n) + 1) / (C(w_n−1) + V)
		 * The smoothed P of a word given context is equal to
		 * the count of the context+word complex plus the smoothing factor, normalized by
		 * the count of the context plus the total amount introduced by smoothing into the whole vocabulary
		 * (Why the count of the context?  Because we're normalizing by the sum of all counts of bigrams beginning with the context.
		 * "(The reader should take a moment to be convinced of this)" -- J&M)
		 */
		// 5 cases:
		Pair<String,String> b = new Pair<String,String>(context,w);
		if (probCounter.containsKey(b)) { // known + known in vocab
			// calculate the smoothed P of that bigram, normalizing appropriately
			return (probCounter.getCount(b)+1.0)/(tokens.getCount(context)+vocabulary.size()); 
		}
		else if (vocabulary.contains(context) && vocabulary.contains(w)) { // known + known out of vocab
			return (tokens.getCount(UNK))/(tokens.getCount(context)+vocabulary.size()); // unigram probability bcause we don't have the bigram.  this is not backoff.
		}
		else if (!vocabulary.contains(context) && !vocabulary.contains(w)) { // unknown + unknown
			return (tokens.getCount(UNK))/(totalBigrams+vocabulary.size()); // ????????????????????????
		}
		else if (!vocabulary.contains(w)) { // known + unknown
			return (tokens.getCount(UNK))/(tokens.getCount(context)+vocabulary.size()); // == known known oov
		}
		else { // unknown + known
			return (tokens.getCount(w)+1.0)/(totalTokens+vocabulary.size());
		}
	}

	/* (non-Javadoc)
	 * @see cs114.langmodel.LanguageModel#getVocabulary()
	 */
	@Override
	public Collection<String> getVocabulary() {
		return vocabulary; // returns the vocabulary of this model
	}

	// generates the next word given a context--in this class, one word.
	private String getNext(String context) {
        double sample = Math.random(); // sets up the random threshold for selection
        double sum = 0.0; // initializes the probability summation for meeting the threshold

        for (Pair<String,String> bigram : probCounter.keySet()) { // look through all bigrams...
        	// and add up the probabilities of each one that starts with the context
        	if (bigram.getFirst() == context){ // if the given context is the first element of the bigram
        		sum += probCounter.getCount(bigram)/tokens.getCount(context); // add the probability of the bigram given the context
        		/*
        		 * (why don't we use getWordProbability() instead?
        		 * Because that method is a diagnostic: it shows the likelihood of a new observation.
        		 * This method needs the likelihood of the bigram within the already-sampled corpus.)
        		 */
        	}
            if (sum > sample) { // if the threshold has been reached
                return bigram.getSecond(); // return the second word of the current bigram
            }
        }
        return LanguageModel.UNK;	
	}
	
	/* (non-Javadoc)
	 * @see cs114.langmodel.LanguageModel#generateSentence()
	 */
	@Override
	public List<String> generateSentence() {
		List<String> s = new ArrayList<String>(); // initialize sentence to be output
		String w = getNext(START);
		while (!w.equals(STOP)){
			s.add(w);
			w = getNext(w);
		}
		return s;
	}
	
	// calling this.getSentenceLogProbability goes directly to the superclass's implementation.

}
