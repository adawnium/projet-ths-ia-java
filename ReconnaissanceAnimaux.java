import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ReconnaissanceAnimaux {

    // --- CLASSE UTILITAIRE POUR STOCKER LES DONNÉES ---
    public static class Dataset {
        public float[][] entrees;
        public float[] resultats;
    }

    // --- 1 & 2. LECTURE, LABELLISATION ET NORMALISATION ---
    public static Dataset chargerEtNormaliser(String repertoireBase) {
        // CORRECTION : On va chercher les fichiers dossier par dossier 
        // pour contourner le fait que la fonction du prof n'est pas récursive
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
            System.err.println("Vérifie que ton terminal est bien ouvert à la racine du projet.");
            return null;
        }

        int nbImages = tousLesFichiers.size();
        
        // On détermine la taille de la première image pour configurer le neurone
        Image premiereImage = new Image(tousLesFichiers.get(0), 0, true);
        int nbPixels = premiereImage.taille();

        Dataset dataset = new Dataset();
        dataset.entrees = new float[nbImages][nbPixels];
        dataset.resultats = new float[nbImages];

        System.out.println("Chargement réussi de " + nbImages + " images depuis " + repertoireBase + "...");

        for (int i = 0; i < nbImages; i++) {
            String chemin = tousLesFichiers.get(i);

            // LABELLISATION : 1.0f pour un chat, 0.0f pour le reste (chien ou sauvage)
            dataset.resultats[i] = chemin.contains("cat") ? 1.0f : 0.0f;

            Image img = new Image(chemin, 0, true);
            int[] pixelsBruts = img.donnees();

            // NORMALISATION : [0, 255] -> [0, 1]
            for (int p = 0; p < nbPixels; p++) {
                dataset.entrees[i][p] = pixelsBruts[p] / 255.0f;
            }
        }
        return dataset;
    }

    // --- 3. MÉLANGE DES DONNÉES (Fisher-Yates) ---
    public static void melangerDataset(Dataset dataset) {
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

    // --- PROGRAMME PRINCIPAL ---
    public static void main(String[] args) {
        
        System.out.println("--- PRÉPARATION DE L'ENTRAÎNEMENT ---");
        // Le code va maintenant chercher automatiquement dans train/cat, train/dog, etc.
        Dataset trainData = chargerEtNormaliser("dataset_animaux/train");
        if (trainData == null) return;
        
        System.out.println("Mélange des données d'entraînement...");
        melangerDataset(trainData);

        System.out.println("\n--- APPRENTISSAGE ---");
        int nbPixels = trainData.entrees[0].length;
        System.out.println("Création d'un neurone avec " + nbPixels + " synapses.");
        
        iNeurone neurone = new NeuroneReLU(nbPixels);
        final float MSElimite = 0.05f; 
        
        System.out.println("Lancement de l'apprentissage (cela peut prendre du temps)...");
        neurone.apprentissage(trainData.entrees, trainData.resultats, MSElimite);

        System.out.println("\n--- TEST DU MODÈLE ---");
        // Le code va maintenant chercher dans test/cat, test/dog, etc.
        Dataset testData = chargerEtNormaliser("dataset_animaux/test");
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