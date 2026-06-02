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
        Image premiereImage = new Image(tousLesFichiers.get(0), 1, true); //peu importe label ou niveau de gris car on lit juste la taille de l'image
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

            int[] pixelsBruts = img.donnees();

            // NORMALISATION : on passe de 0 à 255 à 0 à 1
            for (int p = 0; p < nbPixels; p++) {
                dataset.entrees[i][p] = pixelsBruts[p] / 255.0f;
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
        
        System.out.println("--- PRÉPARATION DE L'ENTRAÎNEMENT ---");
        Dataset trainData = chargerEtLabeliser("dataset_animaux/train");
        if (trainData == null) return;
        
        //on mélange les données
        melangerDataset(trainData);

        System.out.println("\n--- APPRENTISSAGE ---");
        int nbPixels = trainData.entrees[0].length;
        System.out.println("Création d'un neurone avec " + nbPixels + " synapses.");
        
        // iNeurone neurone = new NeuroneReLU(nbPixels);
        iNeurone neurone = new NeuroneSigmoide(nbPixels);
        // iNeurone neurone = new NeuroneHeavyside(nbPixels);
        final float MSElimite = 0.07f; 
        
        System.out.println("Lancement de l'apprentissage...");
        neurone.apprentissage(trainData.entrees, trainData.resultats, MSElimite);

        System.out.println("\n--- TEST DU MODÈLE ---");
        Dataset testData = chargerEtLabeliser("dataset_animaux/test");
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