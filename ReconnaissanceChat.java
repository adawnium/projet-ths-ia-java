import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ReconnaissanceChat {

    // entrees = tableau 2D, ligne = image et colonne = pixel
    // resultats = tableau 1D, ligne = image et valeur = label
    public static class Dataset {
        public float[][] entrees;
        public float[] resultats;
    }

    public static Dataset chargerEtLabeliser(String repertoireBase) {
        List<String> chats = Image.listeFichiers(repertoireBase + "/cat");
        List<String> chiens = Image.listeFichiers(repertoireBase + "/dog");
        List<String> sauvages = Image.listeFichiers(repertoireBase + "/wild");

        // On rassemble tous les chemins d'images dans une seule grande liste
        List<String> tousLesFichiers = new ArrayList<>();
        if (chats != null) tousLesFichiers.addAll(chats);
        if (chiens != null) tousLesFichiers.addAll(chiens);
        if (sauvages != null) tousLesFichiers.addAll(sauvages);
        
        if (tousLesFichiers.isEmpty()) {
            System.err.println("Erreur: Aucun fichier trouvé dans les sous-dossiers de " + repertoireBase);
            return null;
        }

        int nbImages = tousLesFichiers.size();
        
        // On détermine la taille de la première image pour configurer le neurone
        Image premiereImage = new Image(tousLesFichiers.get(0), 1, false); //peu importe label ou niveau de gris car on lit juste la taille de l'image
        int nbPixels = premiereImage.taille();

        Dataset dataset = new Dataset();
        dataset.entrees = new float[nbImages][nbPixels];
        dataset.resultats = new float[nbImages];

        System.out.println("Chargement réussi de " + nbImages + " images depuis " + repertoireBase + "...");

        for (int i = 0; i < nbImages; i++) {
            String chemin = tousLesFichiers.get(i);

            // LABELLISATION : 1 pour un chat, 0 pour le reste (chien ou sauvage)
            int label = chemin.contains("cat") ? 1 : 0; 

            // On construit l'Image avec le bon label et  on le lit avec img.label()
            Image img = new Image(chemin, label, false);
            dataset.resultats[i] = img.label();

            
            // NORMALISATION : on passe de 0 à 255 à 0 à 1
            // int[] pixelsBruts = img.donnees();
            // for (int p = 0; p < nbPixels; p++) {
            //     dataset.entrees[i][p] = pixelsBruts[p] / 255.0f;
            // }

            float[] tsl = ConversionTSL.convertirEnTSL(img);
            for (int p = 0; p < nbPixels; p++) {
                dataset.entrees[i][p] = tsl[p]; // déjà entre 0 et 1
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
        Image premiereImage = new Image(tousLesFichiers.get(0), 1, true);
        int nbPixels = premiereImage.taille();

        Dataset dataset = new Dataset();
        dataset.entrees = new float[nbImages][nbPixels];
        dataset.resultats = new float[nbImages];

        System.out.println("Chargement et calcul FFT de " + nbImages + " images depuis " + repertoireBase + "...");

        for (int i = 0; i < nbImages; i++) {
            String chemin = tousLesFichiers.get(i);
            int label = chemin.contains("cat") ? 1 : 0; 
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

    // algo de fisher yates pour mélanger les données d'entrainement et leur label de facon aléatoire
    public static void melangerDataset(Dataset dataset) {
        System.out.println("Mélange en cours");
        int n = dataset.resultats.length;
        Random random = new Random();
        
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            
            float[] tempEntree = dataset.entrees[i];
            dataset.entrees[i] = dataset.entrees[j];
            dataset.entrees[j] = tempEntree;
            
            float tempResultat = dataset.resultats[i];
            dataset.resultats[i] = dataset.resultats[j];
            dataset.resultats[j] = tempResultat;
        }
    }

    public static void main(String[] args) {
        
        System.out.println("CHARGEMENT DES IMAGES D'ENTRAINEMENT");
        //Dataset trainData = chargerEtLabeliser("dataset_animaux/train");
        Dataset trainData = chargerEtAppliquerFFT("dataset_animaux/train");
        if (trainData == null) return;
        
        //on mélange les données
        melangerDataset(trainData);

        //on limite le nombre d'images (pour accélérer l'apprentissage)
        int limite = Math.min(1000, trainData.resultats.length);
        trainData.entrees = java.util.Arrays.copyOf(trainData.entrees, limite);
        trainData.resultats = java.util.Arrays.copyOf(trainData.resultats, limite);

        System.out.println("\nAPPRENTISSAGE");
        int nbPixels = trainData.entrees[0].length;
        System.out.println("Création d'un neurone avec " + nbPixels + " synapses.");
        
        //choix du modèle de neurone (à décommenter au besoin)
        //iNeurone neurone = new NeuroneReLU(nbPixels);
         iNeurone neurone = new NeuroneSigmoide(nbPixels);
        // iNeurone neurone = new NeuroneHeavyside(nbPixels);
        final float MSElimite = 0.10f; 
        
        System.out.println("Lancement de l'apprentissage :");
        neurone.apprentissage(trainData.entrees, trainData.resultats, MSElimite);

        System.out.println("\nPHASE DE TEST");
        //Dataset testData = chargerEtLabeliser("dataset_animaux/test");
        Dataset testData = chargerEtAppliquerFFT("dataset_animaux/test");

        if (testData == null) return;

        int bonnesReponses = 0;
        int totalTest = testData.resultats.length;

        for (int i = 0; i < totalTest; i++) {
            neurone.metAJour(testData.entrees[i]);
            float certitude = neurone.sortie();

            float prediction = certitude >= 0.5f ? 1.0f : 0.0f;
            
            // System.out.printf("Image de test %d : certitude = %.4f -> %s\n", i, certitude, certitude >= 0.5f ? "Chat" : "Pas Chat");
            
            float vraiLabel = testData.resultats[i];
            if (prediction == vraiLabel) {
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