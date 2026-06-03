import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ReconnaissanceAnimaux {

    public static class Dataset {
        public float[][] entrees;
        public float[] resultats; // label réel : 0=chat, 1=chien, 2=wild
    }

    public static Dataset chargerEtLabeliser(String repertoireBase) {
        List<String> chats    = Image.listeFichiers(repertoireBase + "/cat");
        List<String> chiens   = Image.listeFichiers(repertoireBase + "/dog");
        List<String> sauvages = Image.listeFichiers(repertoireBase + "/wild");

        List<String> tousLesFichiers = new ArrayList<>();
        if (chats    != null) tousLesFichiers.addAll(chats);
        if (chiens   != null) tousLesFichiers.addAll(chiens);
        if (sauvages != null) tousLesFichiers.addAll(sauvages);

        if (tousLesFichiers.isEmpty()) {
            System.err.println("Erreur: aucun fichier trouvé dans " + repertoireBase);
            return null;
        }

        int nbImages = tousLesFichiers.size();
        Image premiereImage = new Image(tousLesFichiers.get(0), 0, true);
        int nbPixels = premiereImage.taille();

        Dataset dataset = new Dataset();
        dataset.entrees   = new float[nbImages][nbPixels];
        dataset.resultats = new float[nbImages];

        System.out.println("Chargement de " + nbImages + " images depuis " + repertoireBase + "...");

        for (int i = 0; i < nbImages; i++) {
            String chemin = tousLesFichiers.get(i);

            int label = chemin.contains("cat")  ? 0
                    : chemin.contains("dog")  ? 1
                    : chemin.contains("wild") ? 2
                    : 3;

            Image img = new Image(chemin, label, true);
            dataset.resultats[i] = img.label();

            int[] pixelsBruts = img.donnees();
            for (int p = 0; p < nbPixels; p++) {
                dataset.entrees[i][p] = pixelsBruts[p] / 255.0f;
            }
        }
        return dataset;
    }

     public static Dataset chargerEtAppliquerFFT(String repertoireBase) {
        List<String> chats = Image.listeFichiers(repertoireBase + "/cat");
        List<String> chiens = Image.listeFichiers(repertoireBase + "/dog");
        List<String> sauvages = Image.listeFichiers(repertoireBase + "/wild");

        List<String> tousLesFichiers = new ArrayList<>();
        if (chats != null) tousLesFichiers.addAll(chats);
        if (chiens != null) tousLesFichiers.addAll(chiens);
        if (sauvages != null) tousLesFichiers.addAll(sauvages);
        
        if (tousLesFichiers.isEmpty()) {
            System.err.println("Erreur: Aucun fichier trouvé dans " + repertoireBase);
            return null;
        }

        int nbImages = tousLesFichiers.size();
        Image premiereImage = new Image(tousLesFichiers.get(0), 0, true);
        int nbPixels = premiereImage.taille();

        Dataset dataset = new Dataset();
        dataset.entrees = new float[nbImages][nbPixels];
        dataset.resultats = new float[nbImages];

        System.out.println("Chargement et calcul FFT de " + nbImages + " images depuis " + repertoireBase + "...");

        for (int i = 0; i < nbImages; i++) {
            String chemin = tousLesFichiers.get(i);
            int label = chemin.contains("cat")  ? 0
          : chemin.contains("dog")  ? 1
          : chemin.contains("wild") ? 2
          : 3;
            Image img = new Image(chemin, label, true);
            dataset.resultats[i] = img.label();

            int[] pixelsBruts = img.donnees();
            Complexe[] signalComplexe = new Complexe[nbPixels];

            // 1. Conversion des pixels en nombres complexes (partie imaginaire = 0)
            for (int p = 0; p < nbPixels; p++) {
                signalComplexe[p] = new ComplexeCartesien(pixelsBruts[p], 0);
            }

            // 2. Application de la transformée de Fourier
            // ATTENTION: Remplacer "appliqueSur" par la méthode exacte de votre fichier FFTCplx 
            // (ex: FFTCplx.fft(), FFTCplx.calcule(), etc.)
            Complexe[] spectreFFT = FFTCplx.appliqueSur(signalComplexe);

            // 3. Extraction du module (amplitude) et recherche du maximum
            float maxAmplitude = 0.0001f; // Évite la division par zéro
            for (int p = 0; p < nbPixels; p++) {
                float amplitude = (float) spectreFFT[p].mod();
                dataset.entrees[i][p] = amplitude;
                
                if (amplitude > maxAmplitude) {
                    maxAmplitude = amplitude;
                }
            }

            // 4. Normalisation "maison" [0, 1] basée sur l'amplitude maximale de l'image
            for (int p = 0; p < nbPixels; p++) {
                dataset.entrees[i][p] = dataset.entrees[i][p] / maxAmplitude;
            }
        }
        return dataset;
    }

    public static void melangerDataset(Dataset dataset) {
        int n = dataset.resultats.length;
        Random random = new Random();
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            float[] tempE = dataset.entrees[i];
            dataset.entrees[i] = dataset.entrees[j];
            dataset.entrees[j] = tempE;
            float tempR = dataset.resultats[i];
            dataset.resultats[i] = dataset.resultats[j];
            dataset.resultats[j] = tempR;
        }
    }

    // Construit un tableau de cibles binaires : 1.0 si l'image appartient à
    // la classe ciblée, 0.0 sinon — puis lance l'apprentissage.
    public static iNeurone entrainerNeurone(Dataset trainData, int classCible, float MSElimite) {
        int nbImages = trainData.resultats.length;
        float[] ciblesBinaires = new float[nbImages];
        for (int i = 0; i < nbImages; i++) {
            ciblesBinaires[i] = (trainData.resultats[i] == classCible) ? 1.0f : 0.0f;
        }
        iNeurone neurone = new NeuroneSigmoide(trainData.entrees[0].length); //sigmoide est le plus efficace
        neurone.apprentissage(trainData.entrees, ciblesBinaires, MSElimite);
        return neurone;
    }

    public static void main(String[] args) {

        System.out.println("--- PRÉPARATION ---");
        Dataset trainData = chargerEtLabeliser("dataset_animaux/train");
        if (trainData == null) return;
        melangerDataset(trainData);

        //limiter le nombre d'images train pour accélérer l'apprentissage
        int limite = Math.min(3000, trainData.resultats.length);
        trainData.entrees = java.util.Arrays.copyOf(trainData.entrees, limite);
        trainData.resultats = java.util.Arrays.copyOf(trainData.resultats, limite);

        System.out.println("\n--- APPRENTISSAGE (One-vs-All) ---");
        final float MSElimite = 0.05f;

        System.out.println("Entraînement neurone chat :");
        iNeurone neuroneChat = entrainerNeurone(trainData, 0, MSElimite);

        System.out.println("Entraînement neurone chien :");
        iNeurone neuroneChien = entrainerNeurone(trainData, 1,MSElimite);

        System.out.println("Entraînement neurone wild :");
        iNeurone neuroneWild = entrainerNeurone(trainData, 2, MSElimite);

        System.out.println("\n--- TEST DU MODÈLE ---");
        Dataset testData = chargerEtLabeliser("dataset_animaux/test");
        if (testData == null) return;

        int bonnesReponses = 0;
        int totalTest = testData.resultats.length;

        for (int i = 0; i < totalTest; i++) {
            float[] entree = testData.entrees[i];

            // Chaque neurone donne sa certitude
            neuroneChat.metAJour(entree);
            neuroneChien.metAJour(entree);
            neuroneWild.metAJour(entree);

            float scoreChat  = neuroneChat.sortie();
            float scoreChien = neuroneChien.sortie();
            float scoreWild  = neuroneWild.sortie();

            // La classe prédite est celle avec le score le plus élevé
            int prediction;
            if (scoreChat >= scoreChien && scoreChat >= scoreWild) {
                prediction = 0;
            } else if (scoreChien >= scoreWild) {
                prediction = 1;
            } else {
                prediction = 2;
            }

            if (prediction == (int) testData.resultats[i]) {
                bonnesReponses++;
            }
        }

        float precision = ((float) bonnesReponses / totalTest) * 100.0f;
        System.out.println("\n========================================");
        System.out.printf("RÉSULTAT FINAL : %.2f%% de précision\n", precision);
        System.out.println("(" + bonnesReponses + " images correctement classées sur " + totalTest + ")");
        System.out.println("========================================");
    }
}