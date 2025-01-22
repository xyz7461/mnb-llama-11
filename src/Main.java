public class Main {

    public static void main(String args[]) throws Exception {
        try {
            MNB_Classification classifier = new MNB_Classification();
            classifier.initializeSets("test_set");
        } catch (Exception e){
            System.out.println(e.getMessage());
        }

    }
}
