package cs114.assignments;

import cs114.langmodel.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SimpleTester {


    public static void testProbabilityDistribution(LanguageModel model, ArrayList<List<String>> sentences) {
        List<List<String>> contexts = new ArrayList<List<String>>();
        contexts.add(new ArrayList<String>(Arrays.asList("A".split(" "))));
        contexts.add(new ArrayList<String>(Arrays.asList("A B".split(" "))));
        contexts.add(new ArrayList<String>(Arrays.asList("D".split(" "))));

        for (int i = 0; i < 5; i++) {
            List<String> randomSentence = new ArrayList<String>(model.generateSentence());
            contexts.add(randomSentence.subList(0, (int) (Math.random() * randomSentence.size())));
        }

        for (List<String> context : contexts) {
            System.out.print("Testing context " + context + " ... ");
            double modelsum = model.checkProbability(context);
            if (Math.abs(1.0 - modelsum) > 1e-6) {
                System.out.println("\nWARNING: probability distribution does not sum up to one. Sum:" + modelsum);
            }
            else {
                System.out.println("GOOD!");
            }
        }
        System.out.println();
        
        for (List<String> sentence : sentences) {
        	System.out.println("Testing sentence " + sentence + ": Prob = " + Math.exp(model.getSentenceLogProbability(sentence)));
        	
        }
    }


    public static void main(String[] args) {
        LanguageModel model;
        model = new ExampleUnigram();
        ArrayList<List<String>> sentences = new ArrayList<List<String>>();
        sentences.add(Arrays.asList("A B B A".split(" ")));
        sentences.add(Arrays.asList("A B B A A A".split(" ")));
        sentences.add(Arrays.asList("B A A B".split(" ")));
        sentences.add(Arrays.asList("A".split(" ")));
        sentences.add(Arrays.asList("A A A".split(" ")));
        sentences.add(Arrays.asList("".split(" ")));
        model.train(sentences);
        
        testProbabilityDistribution(model, sentences);

    }

}
