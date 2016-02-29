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
import cs114.util.Triple;

/**
 * @author clay riley
 * 
 * This LM uses Laplace smoothing and interpolation over trigrams.
 * 
 * TODO This LM's functionality is broken in its current implementation.
 * 
 */
public class TLaplaceTinyInterpolated extends LanguageModel {

	private Counter<Triple<String, String, String>> trigramCounter = new Counter<Triple<String,String,String>>();
	private Counter<Pair<String, String>> bigramCounter = new Counter<Pair<String, String>>();
    private Set<String> vocabulary; // "Keep it secret...
	private Counter<String> tokens = new Counter<String>(); // counter for unigrams
	private double totalTokens;
	// private double totalBigrams;
	private double smoothing = 0.0011;
	
	/* (non-Javadoc)
	 * @see cs114.langmodel.LanguageModel#train(java.util.Collection)
	 */
	@Override
	public void train(Collection<List<String>> trainingSentences) {
		for (List<String> s : trainingSentences) { // for each sentence in training data...
			
			
			
			
			
			// if the sentence is empty (size = 0), only trigram is START+START+STOP
			if (s.size() == 0){
				Triple<String,String,String> empty = new Triple<String,String,String>(START,START,STOP);
				Pair<String,String> bStart = new Pair<String,String>(START,START);
				Pair<String,String> bEnd = new Pair<String,String>(START,STOP);
				trigramCounter.incrementCount(empty, 1.0); // increment trigrams...
				bigramCounter.incrementCount(bStart, 1.0); // ...bigrams...
				bigramCounter.incrementCount(bEnd, 1.0);
				tokens.incrementCount(START, 2.0); // ...and unigrams
				tokens.incrementCount(STOP, 1.0);
			}
			else { 
				// add the start, but not end, to trigram, bigram and unigram counters
				Triple<String,String,String> tStart = new Triple<String,String,String>(START,START,s.get(0));
				Pair<String,String> bStart1 = new Pair<String,String>(START,START);
				Pair<String,String> bStart2 = new Pair<String,String>(START,s.get(0));
				trigramCounter.incrementCount(tStart, 1.0); // increment the 1 observed trigram...
				bigramCounter.incrementCount(bStart1,1.0); // ...the two bigrams...
				bigramCounter.incrementCount(bStart2,1.0);
				// ... and the three unigrams
				tokens.incrementCount(tStart.getFirst(),2.0); // tokens at w-2, w-1
				tokens.incrementCount(tStart.getThird(),1.0); // token at w (the first item in the sentence)
				if (s.size()==1){ // special case if there's only one item
					Triple<String,String,String> t = new Triple<String,String,String>(START,s.get(0),STOP);
					Pair<String,String> b = new Pair<String,String>(s.get(0),STOP);
					trigramCounter.incrementCount(t, 1.0);
					bigramCounter.incrementCount(b, 1.0);
					tokens.incrementCount(STOP, 1.0);
				}
				else if (s.size()==2) { // special case if there are only two items
					Triple<String,String,String> t1 = new Triple<String,String,String>(START,s.get(0),s.get(1));
					Pair<String,String> b1 = new Pair<String,String>(t1.getSecond(),t1.getThird());
					Triple<String,String,String> t2 = new Triple<String,String,String>(s.get(0),s.get(1),STOP);
					Pair<String,String> b2 = new Pair<String,String>(t2.getSecond(),t2.getThird());
					trigramCounter.incrementCount(t1, 1.0);
					trigramCounter.incrementCount(t2, 1.0);
					bigramCounter.incrementCount(b1, 1.0);
					bigramCounter.incrementCount(b2, 1.0);
					tokens.incrementCount(b2.getFirst(), 1.0); // token 2
					tokens.incrementCount(b2.getSecond(), 1.0); // STOP
				}
				else { // more than 2 tokens in sentence --> we can loop through
					Triple<String,String,String> t0 = new Triple<String,String,String>(START,s.get(0),s.get(1)); // the last trigram to contain a START
					Pair<String,String> b0 = new Pair<String,String>(t0.getSecond(),t0.getThird());
					trigramCounter.incrementCount(t0, 1.0);
					bigramCounter.incrementCount(b0, 1.0); // w-1, w (tokens 1 and 2)
					tokens.incrementCount(t0.getThird(), 1.0); // w (token 2)
					for (int i = 2; i<s.size(); i++) { // loop through the sentence
						Triple<String,String,String> t = new Triple<String,String,String>(s.get(i-2),s.get(i-1),s.get(0)); // look 2 back
						Pair<String,String> b = new Pair<String,String>(t.getSecond(),t.getThird());
						trigramCounter.incrementCount(t, 1.0);
						bigramCounter.incrementCount(b, 1.0);
						tokens.incrementCount(t.getThird(), 1.0); // current word
					}
					// once end is reached, add STOP
					Triple<String,String,String> tEnd = new Triple<String,String,String>(s.get(s.size()-2),s.get(s.size()-1),STOP);
					Pair<String,String> bEnd = new Pair<String,String>(tEnd.getSecond(),STOP);
					trigramCounter.incrementCount(tEnd,1.0);
					bigramCounter.incrementCount(bEnd,1.0);
					tokens.incrementCount(STOP, 1.0);
				}
			}
		}
		// getWordProbability implementation has changed to make this unnecessary: 
		// tokens.incrementCount(UNK, smoothing); // add UNK smoothing to unigrams
		totalTokens = tokens.totalCount(); // cache these values!
		// totalBigrams = bigramCounter.totalCount(); // cache these values!
		
		vocabulary = new TreeSet<String>();
		vocabulary.addAll(tokens.keySet()); // set-ify this
		vocabulary = Collections.unmodifiableSet(vocabulary); // ...keep it safe"
				
		// bigramCounter = pc; // = Counters.normalize(pc); // normalizing bigram counts...
	}
	
	
	/*
	 * The following methods simplify the interpolation process.
	 */
	private double tProb(String w2, String w1, String word){ // gets the trigram probability of word
		/*
		 * P(w|context2,context1)
		 * = C(context2,context1,w)+smoothing / C(context2,context1)+V*smoothing
		 * 
		 * P(w|context2,context1)
		 * = C(context1,context2,w)/C(context2,context1)
		 * 
		 */
		Triple<String,String,String> t = new Triple<String,String,String>(w2,w1,word);
		Pair<String,String> b = new Pair<String,String>(w2,w1);
		return (trigramCounter.getCount(t)+smoothing)
				/(bigramCounter.getCount(b)+(vocabulary.size()*smoothing));
	}
	private double bProb(String w1, String word){ // gets the bigram probability of word
		/*
		 * P(w|context) 
		 * = C(context, w)+smoothing / C(context)+V*smoothing
		 */
		Pair<String,String> b = new Pair<String,String>(w1,word);
		return (bigramCounter.getCount(b)+smoothing)
				/(tokens.getCount(w1)+(vocabulary.size()*smoothing));
	}
	private double uProb(String word){ // gets the unigram probability of word, Laplace-smoothed
		/*
		 * P(w) 
		 * = C(w)+smoothing / N+V*smoothing
		 */
		return (tokens.getCount(word)+smoothing)
				/(totalTokens+(vocabulary.size()*smoothing));
	}

	/* (non-Javadoc)
	 * @see cs114.langmodel.LanguageModel#getWordProbability(java.util.List, int)
	 */
	@Override
	public double getWordProbability(List<String> sentence, int index) {		
		// assign context words, looking behind
		String c2;
		String c1;
		if (index == 0){
			c2 = START;
			c1 = START;
		}
		else if (index == 1){
			c2 = START;
			c1 = sentence.get(0);
		}
		else {
			c2 = sentence.get(index-2);
			c1 = sentence.get(index-1);
		}
		// assign word
		String w;
		if (index == sentence.size()){
			w = STOP;
		}
		else {
			w = sentence.get(index);
		}
		/*
		 * Interpolation:
		 * all probability types are used to generate a word's probability;
		 * they are each multiplfied by a lambda factor such that the sum of
		 * the lambdas is one.
		 * 
		 */
		double lambdaT = 0.6; 
		double lambdaB = 0.39;
		double lambdaU = 0.01;
		double pT = lambdaT*tProb(c2,c1,w); // calculate trigram probability
		double bT = lambdaB*bProb(c1,w); // calculate bigram probability
		double uT = lambdaU*uProb(w); // calculate unigram probability
		return pT+bT+uT;
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
        			if (bigram.getSecond() != START){
        				return bigram.getSecond(); // return the second word of the current bigram
        			}
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
