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
import cs114.util.Pair;

/**
 * @author clay riley
 * 
 * This LM uses Laplace plus-one smoothing over bigrams.
 * 
 */
public class BLaplaceKatzTiny extends LanguageModel {

	private Counter<Pair<String, String>> bigramCounter = new Counter<Pair<String, String>>();
    private Set<String> vocabulary; // "Keep it secret...
	private Counter<String> tokens = new Counter<String>(); // counter for unigrams
	private double totalTokens;
	private double smoothing = 0.0011;
	
	public BLaplaceKatzTiny() {
		// Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see cs114.langmodel.LanguageModel#train(java.util.Collection)
	 */
	@Override
	public void train(Collection<List<String>> trainingSentences) {
		for (List<String> s : trainingSentences) { // for each sentence in training data...
			// if the sentence is empty (size = 0), only bigram is START+STOP
			if (s.size() == 0){
				Pair<String,String> empty = new Pair<String,String>(START,STOP);
				bigramCounter.incrementCount(empty, 1.0);
				tokens.incrementCount(START, 1.0);
				tokens.incrementCount(STOP, 1.0);
			}
			else {
				// add the start and end to bigram and unigram counters
				Pair<String,String> start = new Pair<String,String>(START,s.get(0));
				bigramCounter.incrementCount(start,1.0);
				tokens.incrementCount(start.getFirst(), 1.0); // add start (context)
				tokens.incrementCount(start.getSecond(), 1.0); // add first word (word)
				Pair<String,String> end = new Pair<String,String>(s.get(s.size()-1),STOP);
				bigramCounter.incrementCount(end, 1.0);
				tokens.incrementCount(end.getSecond(), 1.0); // only add the end (word)
				/* 
				 * if there is only one word in sentence, all three tokens (START,
				 * w, STOP) and both bigrams (START w, w STOP) have now been added.
				 */
				if (s.size() > 1) { // if there are more than 1 word in the sentence, add the rest
					for (int i = 1; i < s.size(); i++) { // look at all of the rest of the words
						Pair<String,String> b = new Pair<String,String>(s.get(i-1),s.get(i)); // look at the last and the current word
						bigramCounter.incrementCount(b, 1.0);
						// the last was already added to the unigram counter!
						tokens.incrementCount(b.getSecond(), 1.0);
						
					}
				}	
			}
		}
		// getWordProbability implementation has changed to make this unnecessary: tokens.incrementCount(UNK, smoothing); // add UNK smoothing to unigrams
		totalTokens = tokens.totalCount(); // cache this value!
		
		vocabulary = new TreeSet<String>();
		vocabulary.addAll(tokens.keySet()); // set-ify this
		vocabulary = Collections.unmodifiableSet(vocabulary); // ...keep it safe"
				
		// bigramCounter = pc; // = Counters.normalize(pc); // normalizing bigram counts...
	}

	/* (non-Javadoc)
	 * @see cs114.langmodel.LanguageModel#getWordProbability(java.util.List, int)
	 */
	@Override
	public double getWordProbability(List<String> sentence, int index) {		
		// assign context and word, looking behind
		
		
		String context;
		if (index == 0){ // if the index is 0, 
			context = START; // then the context is <S>.
		}
		else { // if the index is anything else, 
			context = sentence.get(index-1); // then the context is the word one back from the index.
		}
		String w;
		if (index == sentence.size()){ // if the index is the sentence size,
			w = STOP; // then the word is </S>.
		}
		else { // if the index is anything else,
			w = sentence.get(index); // then the word is the word at the index.
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
		if (bigramCounter.containsKey(b)) { // known + known in vocab
			// calculate the smoothed P of that bigram, normalizing appropriately
			return (bigramCounter.getCount(b)+smoothing)/(tokens.getCount(context)+vocabulary.size()*smoothing); 
		}
		else if (vocabulary.contains(context) && vocabulary.contains(w)) { // known + known out of vocab
			return (tokens.getCount(UNK)+smoothing)/(tokens.getCount(context)+vocabulary.size()*smoothing); // unigram probability bcause we don't have the bigram.  this is not backoff.
		}
		else if (!vocabulary.contains(context) && !vocabulary.contains(w)) { // unknown + unknown
			return (tokens.getCount(UNK)+smoothing)/(totalTokens+vocabulary.size()*smoothing); // 
		}
		else if (!vocabulary.contains(w)) { // known + unknown
			return (tokens.getCount(UNK)+smoothing)/(tokens.getCount(context)+vocabulary.size()*smoothing); // == known known oov
		}
		else { // unknown + known
			return (tokens.getCount(w)+smoothing)/(totalTokens+vocabulary.size()*smoothing);
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

        for (Pair<String,String> bigram : bigramCounter.keySet()) { // look through all bigrams...
        	// and add up the probabilities of each one that starts with the context
        	if (bigram.getFirst().equals(context)){ // if the given context is the first element of the bigram
        		sum += ((bigramCounter.getCount(bigram))/(tokens.getCount(context))); // add the probability of the bigram given the context
        		/*
        		 * (why don't we use getWordProbability() instead?
        		 * Because that method is a diagnostic: it shows the likelihood of a new observation.
        		 * This method needs the likelihood of the bigram within the already-sampled corpus.)
        		 */
        		if (sum > sample) { // if the threshold has been reached
        			return bigram.getSecond(); // return the second word of the current bigram
        		}
        	}
        }
        return LanguageModel.STOP; // if we haven't returned anything, return STOP.
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
