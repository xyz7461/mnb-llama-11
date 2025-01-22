import java.util.HashMap;

public class TestDoc {
    private final String trueClass;
    private final HashMap<String, Integer> wordFreqs;

    public TestDoc(String docClass, HashMap<String, Integer> freqOfWords){
        trueClass = docClass;
        wordFreqs = freqOfWords;
    }

    public HashMap<String, Integer> getWordFreqs() {
        return wordFreqs;
    }

    public String getTrueClass() {
        return trueClass;
    }

    public String toString(){
        return trueClass + ": " + wordFreqs;
    }
}
