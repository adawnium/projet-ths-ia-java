# projet-ths-ia-java

Par soucis de facilité (et aussi de bug Java inrésolvables), tous les fichiers sont à la racine du projet.

Avant de compiler et lancer un fichier, veuillez importer le dossier nommé "dataset_animaux" à la racine du projet. 
=> dataset_animaux/ doit contenir les deux sous-dossiers test/ et train/ contenant eux-même chacun cat/ dog/ et wild/

## Commandes
Compiler les fichiers : javac *.java
Lancer le programme : java ReconnaissanceChat.java


Le fichier __ReconnaissanceChat.java__ est le fichier de base du projet. Le neurone fonctionne de manière binaire : soit chat, soit pas chat


Le fichier permet de tester l'apprentissage du neurone avec diverses options telles que :
- avec ou sans mélange de données
- avec ou sans normalisation
- en RGB/TSL/Noir & Blanc
- le type de neurone utilisé (Heavyside, ReLU, Sigmoïde)

Pour cela, il suffit de décommenter le bout de code non voulu