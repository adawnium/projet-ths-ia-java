# projet-ths-ia-java

Projet Java d'exploration d'un perceptron simple pour la reconnaissance binaire d'images (chat / pas chat).

Remarques générales
- Pour simplicité (et contournement de certains bugs Java), tous les fichiers .java sont placés à la racine du projet.

Organisation du dataset
- Placez un dossier nommé `dataset_animaux/` à la racine du projet. Il doit contenir deux sous-dossiers :
	- `train/` — images d'entraînement
	- `test/`  — images de test
	Chaque sous-dossier doit contenir les classes : `cat/`, `dog/`, `wild/` (ou autres selon vos besoins).

Compilation et exécution
- Compiler tous les fichiers :
	javac *.java
- Lancer le programme principal :
	java ReconnaissanceChat

Fichier principal
- `ReconnaissanceChat.java` contient le flux principal d'apprentissage et d'évaluation. Le modèle implémente une classification binaire (chat / pas chat).
- `ReconnaissanceAnimaux.java` permet de lancer une classification à 3 classes (chats, chiens, animaux sauvages) 

Options et variantes
Le code permet d'activer ou désactiver, via le fichier source, plusieurs options :
- mélange (shuffle) ou non des données
- normalisation ou non des entrées
- représentation des images : RGB, TSL (HSV) ou niveaux de gris
- type d'activation du neurone : Heaviside (seuil), ReLU, Sigmoïde

Usage rapide
- Ouvrez `ReconnaissanceChat.java` et décommentez les blocs correspondant aux options souhaitées (mélange, normalisation, type d'activation, etc.).
- Recompilez et exécutez.

Conseils
- Utilisez des images de taille cohérente (redimensionnez si nécessaire) pour de meilleurs résultats.
- Commencez par un petit jeu d'entraînement pour vérifier que le pipeline fonctionne, puis augmentez les données.

Licence et contact
- Projet éducatif — pas de garantie.
