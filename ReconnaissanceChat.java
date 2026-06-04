import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ReconnaissanceChat {

    // entrees = tableau 2D, ligne = image et colonne = pixel
    // resultats = tableau 1D, ligne = image et valeur = label
    public static class Dataset {
        public float[][] entrees;
        public float[] resultats;
        public float valeurMaxFFT; // pour stocker la valeur maximale de l'espace fréquentiel (pour la normalisation)
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

            
            // NORMALISATION : on divise tous les pixels par 255 pour les ramener entre 0 et 1
            int[] pixelsBruts = img.donnees();
            for (int p = 0; p < nbPixels; p++) {
                dataset.entrees[i][p] = pixelsBruts[p] / 255.0f;
            }

            // Pour TSL
            // float[] tsl = ConversionTSL.convertirEnTSL(img);
            // for (int p = 0; p < nbPixels; p++) {
            //     dataset.entrees[i][p] = tsl[p]; // déjà entre 0 et 1
            // }
        }
        return dataset;
    }

    public static Dataset chargerEtAppliquerFFT(String repertoireBase, float maxForce) {
        List<String> chats = Image.listeFichiers(repertoireBase + "/cat");
        List<String> chiens = Image.listeFichiers(repertoireBase + "/dog");
        List<String> sauvages = Image.listeFichiers(repertoireBase + "/wild");

        List<String> tousLesFichiers = new ArrayList<>();
        if (chats != null) tousLesFichiers.addAll(chats);
        if (chiens != null) tousLesFichiers.addAll(chiens);
        if (sauvages != null) tousLesFichiers.addAll(sauvages);
        
        if (tousLesFichiers.isEmpty()) return null;

        int nbImages = tousLesFichiers.size();
        
        Image premiereImage = new Image(tousLesFichiers.get(0), 0, true);
        int nbPixels = premiereImage.taille();
        int tailleCote = (int) Math.sqrt(nbPixels); 

        Dataset dataset = new Dataset();
        dataset.entrees = new float[nbImages][nbPixels];
        dataset.resultats = new float[nbImages];

        float[][] modulesFFTBruts = new float[nbImages][nbPixels];
        
        float maxGlobalFFT = maxForce; 
        boolean doitCalculerMax = (maxForce <= 0.0f);


        for (int i = 0; i < nbImages; i++) {
            String chemin = tousLesFichiers.get(i);
            Image img = new Image(chemin, 0, true); 
            int[] pixelsBruts = img.donnees();


            Complexe[][] matrice2D = new Complexe[tailleCote][tailleCote];
            for (int y = 0; y < tailleCote; y++) {
                for (int x = 0; x < tailleCote; x++) {
                    matrice2D[y][x] = new ComplexeCartesien(pixelsBruts[y * tailleCote + x], 0.0);
                }
            }


            //on fait une fft 2d en appliquant d'abord la fft 1d sur les lignes, puis sur les colonnes
            for (int y = 0; y < tailleCote; y++) {
                //fft 1d sur ligne
                matrice2D[y] = FFTCplx.appliqueSur(matrice2D[y]);
            }

            for (int x = 0; x < tailleCote; x++) {
                Complexe[] colonne = new Complexe[tailleCote];
                for (int y = 0; y < tailleCote; y++) {
                    colonne[y] = matrice2D[y][x];
                }
                
                // fft 1d sur colonne
                colonne = FFTCplx.appliqueSur(colonne);
                
                for (int y = 0; y < tailleCote; y++) {
                    matrice2D[y][x] = colonne[y];
                }
            }

            for (int y = 0; y < tailleCote; y++) {
                for (int x = 0; x < tailleCote; x++) {
                    float mod = (float) matrice2D[y][x].mod();

                    if (x == 0 && y == 0) {
                        mod = 0.0f;
                    } else {
                        mod = (float) Math.log(1.0 + mod);
                    }

                    int indexPlat = y * tailleCote + x;
                    modulesFFTBruts[i][indexPlat] = mod;
                    
                    if (doitCalculerMax && mod > maxGlobalFFT) {
                        maxGlobalFFT = mod;
                    }
                }
            }

            dataset.resultats[i] = chemin.contains("cat") ? 1.0f : 0.0f;
        }

        dataset.valeurMaxFFT = maxGlobalFFT;

        if (maxGlobalFFT > 0) {
            for (int i = 0; i < nbImages; i++) {
                for (int j = 0; j < nbPixels; j++) {
                    dataset.entrees[i][j] = modulesFFTBruts[i][j] / maxGlobalFFT;
                }
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

    public static void main(String[] args) {
        
        System.out.println("CHARGEMENT DES IMAGES D'ENTRAINEMENT");
        // Dataset trainData = chargerEtLabeliser("dataset_animaux/train");
        Dataset trainData = chargerEtAppliquerFFT("dataset_animaux/train", 0.0f); // 0.0f car on force le calcul de la valeur max pour la normalisation
        if (trainData == null) return;

        float maxValeurFFTTrain = trainData.valeurMaxFFT; 
        
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
        iNeurone neurone = new NeuroneSigmoide(nbPixels);
        // iNeurone neurone = new NeuroneReLU(nbPixels);
        // iNeurone neurone = new NeuroneHeavyside(nbPixels);
        final float MSElimite = 0.005f; 
        
        System.out.println("Lancement de l'apprentissage :");
        neurone.apprentissage(trainData.entrees, trainData.resultats, MSElimite);

        System.out.println("\nPHASE DE TEST");
        // Dataset testData = chargerEtLabeliser("dataset_animaux/test");
        Dataset testData = chargerEtAppliquerFFT("dataset_animaux/test", maxValeurFFTTrain);

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
        System.out.printf("RÉSULTAT FINAL : %.2f%% d'accuracy'\n", precision);
        System.out.println("(" + bonnesReponses + " images correctement classées sur " + totalTest + ")");
        System.out.println("========================================");
    }
}