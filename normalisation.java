import java.util.List;

public class Normalisation {

    /**
     * Normalise les pixels d'une image entre 0.0 et 1.0
     * en divisant chaque valeur par 255.
     */
    public static float[] normaliser(Image image) {
        int[] donnees = image.donnees();
        float[] normalise = new float[donnees.length];

        for (int i = 0; i < donnees.length; i++) {
            normalise[i] = donnees[i] / 255.0f;
        }

        return normalise;
    }

    public static void main(String[] args) {

        // Dossiers et leurs labels
        String[] dossiers = {"cat", "dog", "wild"};
        int[]    labels   = {1,      0,     0};
        String[] noms     = {"CHAT", "CHIEN", "WILD"};

        for (int k = 0; k < dossiers.length; k++) {

            // Récupère tous les fichiers du dossier automatiquement
            List<String> fichiers = Image.listeFichiers(dossiers[k]);

            for (String chemin : fichiers) {

                System.out.println("\n=============================");
                System.out.println("Categorie : " + noms[k]);
                System.out.println("Fichier   : " + chemin);

                Image img = new Image(chemin, labels[k], true);
                img.afficheMetadonnees();

                // 5 pixels AVANT normalisation
                System.out.println("Avant normalisation :");
                int[] donnees = img.donnees();
                for (int i = 0; i < 5; i++) {
                    System.out.printf("  pixel[%d] = %d\n", i, donnees[i]);
                }

                // Normaliser
                float[] normalise = normaliser(img);

                // 5 pixels APRES normalisation
                System.out.println("Apres normalisation :");
                for (int i = 0; i < 5; i++) {
                    System.out.printf("  pixel[%d] = %.4f\n", i, normalise[i]);
                }

                // Vérifier que tout est bien entre 0 et 1
                float min = normalise[0], max = normalise[0];
                for (float v : normalise) {
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
                System.out.printf("Min=%.4f | Max=%.4f\n", min, max);
            }
        }
    }
}
