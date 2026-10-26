"""Télécharge (à la construction de l'image) le modèle de plongements de phrases, à une révision figée."""

from sentence_transformers import SentenceTransformer

MODELE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
modele = SentenceTransformer(MODELE)
print("Modèle prêt :", MODELE, "dimension", modele.get_sentence_embedding_dimension())
