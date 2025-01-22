import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

import static java.lang.Character.*;


public class MNB_Classification {

    HashMap<String,HashMap<String, Integer>> trainingSet = new HashMap<>();
    ArrayList<TestDoc> testSet = new ArrayList<>();
    HashSet<String> vocabulary = new HashSet<>();
    StopWords stopWordRemover = new StopWords();
    private int counter = 0;

    public void initializeSets(String pathToFiles) throws Exception {
        processDocs(pathToFiles);
        System.out.println("Training set: " + trainingSet);
        System.out.println("Test set: " + testSet);
    }

    private void processDocs(String pathToFiles) throws Exception {
        File dataFolder = new File(pathToFiles);
        File[] classFolders = dataFolder.listFiles();
        if (classFolders == null){
            throw new Exception("provided path is not a directory");
        }
        else if (classFolders.length == 0){
            throw new Exception("no class folders found in provided directory");
        }

        File[] classFiles;
        for (File classFolder : classFolders) {
            trainingSet.put(classFolder.getName(), new HashMap<>());
            classFiles = classFolder.listFiles();
            if (classFiles == null){
                System.out.println(classFolder.getName());
                throw new Exception("Loose file in parent directory for class folders");
            }else if (classFiles.length == 0){
                throw new Exception("No class files found in class folder: " + classFolder.getName());
            }
            for (File classFile : classFiles){
                if (putInTraining()){
                    readTrainingFile(classFile, classFolder.getName());
                } else{
                    readTestFile(classFile, classFolder.getName());
                }
            }
        }
    }

    private void readTrainingFile(File classFile, String className) throws Exception {
        if (!classFile.canRead()){
            throw new Exception("File " + classFile + " cannot be read");
        }
        Scanner docScan = new Scanner(classFile);
        // while (true){
        //     if (docScan.nextLine().contains("Lines:")){
        //         break;
        //     }
        //     if (!docScan.hasNext()){
        //         throw new Exception("End of header not found: " + className + "," + classFile.getName());
        //     }
        // }

        String procWord;
        docScan.useDelimiter("\\s|[.]|-");
        while(docScan.hasNext()){
            procWord = processWord(docScan.next());
            if (!procWord.isEmpty()){
                trainingSet.get(className).merge(procWord, 1, Integer::sum);
                vocabulary.add(procWord);
            }
        }
        docScan.close();
    }

    private void readTestFile(File classFile, String className) throws Exception {
        if (!classFile.canRead()){
            throw new Exception("File " + classFile + " cannot be read");
        }
        Scanner docScan = new Scanner(classFile);
        // while (true){
        //     if (docScan.nextLine().contains("Lines:")){
        //         break;
        //     }
        //     if (!docScan.hasNext()){
        //         throw new Exception(" not found: " + className + "," + classFile.getName());
        //     }
        // }

        HashMap<String,Integer> wordFreq = new HashMap<>();
        String procWord;
        while(docScan.hasNext()){
            procWord = processWord(docScan.next());
            if (!procWord.isEmpty()){
                wordFreq.merge(procWord, 1, Integer::sum);
            }
        }
        docScan.close();
        testSet.add(new TestDoc(className,wordFreq));
    }



    private boolean putInTraining(){ // TODO update this to be truly random when not using test data
        counter += 1;
        if (counter == 3){
            return false;
        }
        if (counter == 5){
            counter = 0;
        }
        return true;
    }

    private String processWord(String word){
        //TODO break up by hyphens and periods
        if (stopWordRemover.contains(word)){
            return "";
        }
        if (word.contains("@")){
            return "";
        }

        boolean alphaFound = false;
        StringBuilder procWord = new StringBuilder(word.toLowerCase().trim());
        //getting rid of emails
        //remove ending punctuation and check for a letter
        for (int i = (procWord.length() - 1); i >=0 ; i--){
            if (isAlphabetic(procWord.charAt(i))){
                alphaFound = true;
                break;
            }
            if (!isDigit(procWord.charAt(i))){
                procWord.deleteCharAt(i);
            }
        }
        if (!alphaFound){
            return "";
        }
        //remove starting punctuation
        while (!isLetterOrDigit(procWord.charAt(0))){
            procWord.deleteCharAt(0);
        }

        //check if stopword
        if (stopWordRemover.contains(procWord.toString())){
            return "";
        }

        //System.out.print(procWord + " ");
        return procWord.toString();
    }


    private void label(){}
}
