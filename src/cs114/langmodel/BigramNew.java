package cs114.langmodel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import cs114.util.Counter;
import cs114.util.Counters;
import cs114.util.PriorityQueue;
import cs114.util.Pair;

public class BigramNew extends LanguageModel {

	private Counter<Pair<String,String>> bProbs;
	private Counter<String> uProbs;
	private Set<String> vocab;
	private double totalUs;
	private double totalBs;
	
	@Override
	public void train(Collection<List<String>> trainingSentences) {
		Counter<Pair<String,String>> bCounts = new Counter<Pair<String,String>>();
		Counter<String> uCounts = new Counter<String>();
		for (List<String> s : trainingSentences) {
			if (s.size()==0) {
				Pair<String,String> b = new Pair<String,String>(START,STOP);
				bCounts.incrementCount(b, 1.0);
				uCounts.incrementCount(START, 1.0);
				uCounts.incrementCount(STOP, 1.0);
			} else if (s.size()==1) {
				Pair<String,String> b1 = new Pair<String,String>(START,s.get(0));
				Pair<String,String> b2 = new Pair<String,String>(s.get(0),STOP);
				bCounts.incrementCount(b1, 1.0);
				bCounts.incrementCount(b2,1.0);
				uCounts.incrementCount(START, 1.0);
				uCounts.incrementCount(b2.getFirst(),1.0);
				uCounts.incrementCount(STOP, 1.0);
			} else {
				Pair<String,String> b1 = new Pair<String,String>(START,s.get(0));
				Pair<String,String> b2 = new Pair<String,String>(s.get(s.size()-1),STOP);
				bCounts.incrementCount(b1, 1.0);
				bCounts.incrementCount(b2,1.0);
				uCounts.incrementCount(START, 1.0);
				uCounts.incrementCount(b2.getFirst(),1.0);
				uCounts.incrementCount(STOP, 1.0);
				for (int i=1; i<s.size(); i++) {
					Pair<String,String> b = new Pair<String,String>(s.get(i-1),s.get(i));
					bCounts.incrementCount(b, 1.0);
					uCounts.incrementCount(b.getFirst(), 1.0);
				}
			}
		}
		uCounts.incrementCount(UNK,1.0);
		
		vocab = new TreeSet<String>();
		vocab.addAll(uCounts.keySet());
		vocab = Collections.unmodifiableSet(vocab);
		
		totalUs = uCounts.totalCount();
		totalBs = bCounts.totalCount();
		
		uProbs = Counters.normalize(uCounts);
		bProbs = Counters.normalize(bCounts);
	}

	@Override
	public double getWordProbability(List<String> sentence, int index) {
		// TODO Auto-generated method stub
		if (index == sentence.size()) {
			return 0.0;
		}else{
			return 0.0;
		}
		/*

		if (index == sentence.size()) {
			return probCounter.getCount(STOP);
		}else{
			String word = sentence.get(index);
			if (!probCounter.containsKey(word)){
				return probCounter.getCount(UNK);
			}else {
				return probCounter.getCount(word);
			}
		}
		*/
		
	}

	@Override
	public Collection<String> getVocabulary() {
		return vocab;
	}

	@Override
	public List<String> generateSentence() {
		// TODO Auto-generated method stub
		return null;
	}

}
