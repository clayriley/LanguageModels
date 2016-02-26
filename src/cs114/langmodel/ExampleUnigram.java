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

public class ExampleUnigram extends LanguageModel {

    private Counter<String> probCounter;
    private Set<String> vocabulary;

	public ExampleUnigram() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public void train(Collection<List<String>> trainingSentences) {
		// TODO Auto-generated method stub
		Counter<String> wordCounter = new Counter<String>();
        for (List<String> sentence : trainingSentences) {
            for (String word : sentence) {
                wordCounter.incrementCount(word, 1.0);
            }
            wordCounter.incrementCount(STOP, 1.0);
        }

        wordCounter.incrementCount(UNK, 1.0);
        vocabulary = new TreeSet<String>();
        vocabulary.addAll(wordCounter.keySet());
        vocabulary = Collections.unmodifiableSet(vocabulary);
        
        probCounter = Counters.normalize(wordCounter);
	}

	@Override
	public double getWordProbability(List<String> sentence, int index) {
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
	}

	@Override
	public Collection<String> getVocabulary() {
	    return vocabulary;
	}
	
	private String generateWord() {
        double sample = Math.random();
        double sum = 0.0;

        for (String word : probCounter.keySet()) {
            sum += probCounter.getCount(word);
            if (sum > sample) {
                return word;
            }
        }
        return LanguageModel.UNK;	
	}

	@Override
	public List<String> generateSentence() {
		// TODO Auto-generated method stub
        List<String> sentence = new ArrayList<String>();
        String word;
        do {
            word = generateWord();
            sentence.add(word);
        }
        while (!word.equals(STOP));
        return sentence;
	}

}
