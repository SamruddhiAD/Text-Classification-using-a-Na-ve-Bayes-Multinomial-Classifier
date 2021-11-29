import java.util.*;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
/**
 * ISTE-612-2205 Lab #4
 * Samruddhi Deshpande
 * Date
 */

public class NBClassifier {
	private HashMap<Integer,String> trainingDocs;         //training data
	private int[] trainingClasses;         //training class values
   	private int numClasses=2;
	private int[] classDocCounts;          //number of docs per class 
   	private String[] classStrings;         //concatenated string for a given class 
	private int[] classTokenCounts;        //total number of tokens per class 
	private HashMap<String,Double>[] condProb; //term conditional prob
	private HashSet<String> vocabulary;    //entire vocabulary
	public HashMap<Integer, String> docs;
    public static HashMap<Integer, String> TestDocs;
    public HashMap<Integer, String> fileMap;
	public int[] testingclasses;	
	/**
	 * Build a Naive Bayes classifier using a training document set
	 * @param trainDataFolder the training document folder
	 */
	@SuppressWarnings("unchecked")
	public NBClassifier(String trainDataFolder)
	{
		preprocess(trainDataFolder);
		trainingDocs = docs;
		
		  classDocCounts = new int[numClasses];  // # of documents in each class
		  classStrings = new String[numClasses]; // concatenated all string in each class
		  classTokenCounts = new int[numClasses];// # of token counts in each class
		
		condProb = new HashMap[numClasses];    // a set of conditional prob of all terms in each class
		  vocabulary = new HashSet<String>();
		
		//initialization of classStrings & cond probabilities for each class
		  for(int i=0;i<numClasses;i++){
			  classStrings[i] = "";
			  condProb[i] = new HashMap<String,Double>();
		  }
		
		//populate classDocCounts & classStrings
		  for(int i=0;i<trainingClasses.length;i++){
			  classDocCounts[trainingClasses[i]]++;
			  classStrings[trainingClasses[i]] += (trainingDocs.get(i) + " ");
		  }
		
		//tf calculations
		  for(int i=0;i<numClasses;i++){
			  String[] tokens = classStrings[i].split("[\" ()_,?:;%&-]*'+");
			  classTokenCounts[i] = tokens.length;
			  for(String token:tokens){
				  vocabulary.add(token);
				  if(condProb[i].containsKey(token)){
					  double count = condProb[i].get(token);
					  condProb[i].put(token, count+1);
				  }
				  else
					  condProb[i].put(token, 1.0);
			  }
		  }
		  
		// calculate cond prob with Laplace Smoothing
		for(int i=0;i<numClasses;i++){
			  Iterator<Map.Entry<String, Double>> iterator = condProb[i].entrySet().iterator();
			  int vSize = vocabulary.size();
			  while(iterator.hasNext())
			  {
				  Map.Entry<String, Double> entry = iterator.next();
				  String token = entry.getKey();
				  Double count = entry.getValue();
			  Double prob = (count+1)/(classTokenCounts[i]+vSize);
			  condProb[i].put(token, prob);
			  }
			  //System.out.println(condProb[i]);
		  }
		
	}
	
	/**
	 * Classify a test doc
	 * @param doc test doc
	 * @return class label
	 */
	public int classify(String doc){


		int label = 0;
      // add code here
      int vSize = vocabulary.size();
      double[] score = new double[numClasses];
      
      for(int i=0;i<score.length;i++) {
         score[i] = classDocCounts[i] * 1.0/trainingDocs.size(); // prior probability     
      }
      
      String[] tokens = doc.split(" ");
      
      for(int i=0;i<numClasses;i++) {
         for(String token:tokens) {
            if(condProb[i].containsKey(token))
               //score[i] *= condProb[i].get(token);
               score[i] += Math.log10(condProb[i].get(token));

            else
               //score[i] *= (1.0/(classTokenCounts[i] + vSize));
               score[i] += Math.log10((1.0/(classTokenCounts[i] + vSize)));
         }
      }
      
      //argsMax
      
      double maxScore = score[0];
      //System.out.println("score[0]: "+ score[0]);
      for(int i=0;i<score.length;i++) {
          //System.out.println("score[: "+ i + "]: " + score[i]);

          if(score[i] > maxScore) {
            maxScore = score[i];
            label = i;
          }
      }
           
		return label;
	}
	
	public  String readFileAsString(String fileName) throws Exception
    { 
      String data = ""; 
      data = new String(Files.readAllBytes(Paths.get(fileName))); 
      return data; 
    }
	/**
	 * Load the training documents
	 * @param trainDataFolder
	 */
	public void preprocess(String trainDataFolder)
	{
		

		File train_pos = new File(trainDataFolder + "/pos");
        File train_neg = new File(trainDataFolder + "/neg");
		ArrayList<File> collect = new ArrayList<File>();
		collect.add(train_pos);
		collect.add(train_neg);

        docs = new HashMap<Integer, String>();
        fileMap = new HashMap<Integer, String>();
        int i = 0;
        trainingClasses = new int[3000];
        try {
			for(File c:collect){
            for (File file : c.listFiles()) {
				String filename="";
					if(collect.indexOf(c)==0){
						 filename=trainDataFolder+"/pos/"+(file.getName());
						}
						if(collect.indexOf(c)==1){
						  filename=trainDataFolder+"/neg/"+(file.getName());
						}


					String content = readFileAsString(filename);
                    docs.put(i, content.toLowerCase());
					fileMap.put(i, file.getName());
					if(collect.indexOf(c)==0){
						trainingClasses[i] = 1;
					}
					if(collect.indexOf(c)==1){
						trainingClasses[i] = 0;
						}
                    i++;
                
			}
			}
		}catch (Exception ioe) {
            ioe.printStackTrace();
        }
	}
	
	/**
	 *  Classify a set of testing documents and report the accuracy
	 * @param testDataFolder fold that contains the testing documents
	 * @return classification accuracy
	 */
	public double classifyAll(String testDataFolder)
	{
		
        File test_pos = new File(testDataFolder + "/pos");
		File test_neg = new File(testDataFolder + "/neg");
		ArrayList<File> collecttest = new ArrayList<File>();
		collecttest.add(test_pos);
		collecttest.add(test_neg);

		TestDocs = new HashMap<Integer, String>();
        fileMap = new HashMap<Integer, String>();
        int i = 0;
        testingclasses = new int[3000];
        try {
			for(File c:collecttest){
            for (File file : c.listFiles()) {
				String filename="";
				if(collecttest.indexOf(c)==0){
					 filename=testDataFolder+"/pos/"+(file.getName());
					}
					if(collecttest.indexOf(c)==1){
					  filename=testDataFolder+"/neg/"+(file.getName());
					}
					String content = readFileAsString(filename);
                    TestDocs.put(i, content.toLowerCase());
					fileMap.put(i, file.getName());
					if(collecttest.indexOf(c)==0){
						testingclasses[i] = 1;
					}
					if(collecttest.indexOf(c)==1){
						testingclasses[i] = 0;
						}
                    i++;
                
			}
			}
		}catch (Exception ioe) {
            ioe.printStackTrace();
		}

		float Tpos = 0;
		float Tneg = 0;
		float Fpos = 0;
		float Fneg = 0;
		int correctlyClassified = 0;
		double accuracy;
		for (Map.Entry<Integer, String> testDoc : TestDocs.entrySet()) {
			int result = classify(testDoc.getValue());
			if (trainingClasses[testDoc.getKey()] == 1 && result == trainingClasses[testDoc.getKey()]) {
				Tpos++;
			} else if (trainingClasses[testDoc.getKey()] == 0 && result == trainingClasses[testDoc.getKey()]) {
				Tneg++;
			} else if (trainingClasses[testDoc.getKey()] == 0 && result != trainingClasses[testDoc.getKey()]) {
				Fpos++;
			} else if (trainingClasses[testDoc.getKey()] == 1 && result != trainingClasses[testDoc.getKey()]) {
				Fneg++;
			}
		}
		correctlyClassified = (int) Tpos + (int) Tneg;
		accuracy = (Tpos + Tneg) / (Tpos + Tneg + Fpos + Fneg);

		try{
		System.out.println("Classifying individual testing documents:");
		String filename=testDataFolder+"/pos/cv989_15824.txt";
		String content = readFileAsString(filename);
		int out = classify(content);
		System.out.println("cv989_15824.txt is :");
		System.out.println(out == 1 ? "Positive" : "Negative");
		}catch (Exception ioe) {
            ioe.printStackTrace();
		}

		System.out.println("Correctly classified " + correctlyClassified + " out of " + TestDocs.size());
		return accuracy;
	}
	
	
	public static void main(String[] args)
	{	
		NBClassifier nb = new NBClassifier("data/train");
		Double acc = nb.classifyAll("data/test");
		System.out.println("Accuracy: " +acc);

	}
}
