public class ConversionTSL {

    /**
     * Convertit une image RGB en TSL (Teinte, Saturation, Luminosité).
     * Les valeurs TSL sont normalisées entre 0.0 et 1.0.
     *
     * L'image doit avoir été chargée avec niveauxDeGris=false (RGB).
     * Le tableau retourné est de même taille que donnees() : 3 * hauteur * largeur,
     * avec pour chaque pixel : [T, S, L] dans cet ordre.
     */
    public static float[] convertirEnTSL(Image image) {
        int[] donnees = image.donnees();
        int nbPixels = donnees.length / 3; // car RGB : 3 valeurs par pixel
        float[] tsl = new float[donnees.length]; // même taille : T, S, L par pixel
 
        for (int i = 0; i < nbPixels; i++) {
            // Récupérer R, G, B normalisés entre 0 et 1
            float r = donnees[3 * i + 0] / 255.0f;
            float g = donnees[3 * i + 1] / 255.0f;
            float b = donnees[3 * i + 2] / 255.0f;
 
            float max = Math.max(r, Math.max(g, b));
            float min = Math.min(r, Math.min(g, b));
            float delta = max - min;
 
            // --- Luminosité L ---
            float L = (max + min) / 2.0f;
 
            // --- Saturation S ---
            float S = 0.0f;
            if (delta != 0.0f) {
                S = delta / (1.0f - Math.abs(2.0f * L - 1.0f));
            }
 
            // --- Teinte T (normalisée entre 0 et 1, 0=rouge, 1/6=jaune, etc.) ---
            float T = 0.0f;
            if (delta != 0.0f) {
                if (max == r) {
                    T = ((g - b) / delta) % 6.0f;
                } else if (max == g) {
                    T = (b - r) / delta + 2.0f;
                } else {
                    T = (r - g) / delta + 4.0f;
                }
                T = T / 6.0f; // ramener entre 0 et 1
                if (T < 0.0f) T += 1.0f; // corriger les valeurs négatives
            }
 
            tsl[3 * i + 0] = T;
            tsl[3 * i + 1] = S;
            tsl[3 * i + 2] = L;
        }
 
        return tsl;
    }
}