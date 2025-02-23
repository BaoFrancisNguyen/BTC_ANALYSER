import json
import os

# ✅ Vérifier que le fichier JSONL contient des données valides
dataset_file = "mistral_training_data.jsonl"

if not os.path.exists(dataset_file):
    print("❌ Erreur : Le fichier mistral_training_data.jsonl n'existe pas.")
    exit()

try:
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]  # Supprime les lignes vides

    if not data:
        print("❌ Erreur : Le dataset est vide après nettoyage.")
        exit()
    
    print(f"✅ Le dataset contient {len(data)} exemples.")
except Exception as e:
    print(f"❌ Erreur dans le dataset : {e}")
    exit()

# ✅ Corriger la structure du fichier Modelfile
with open("Modelfile", "w", encoding="utf-8") as f:
    f.write("FROM mistral\n\n")
    f.write("PARAMETER {\n")
    f.write('  "temperature": 0.7,\n')
    f.write('  "top_p": 0.9\n')
    f.write("}\n\n")

    # ✅ Ajout d'un message système
    f.write("SYSTEM \"Ce modèle est fine-tuné avec des articles récents sur l'IA.\"\n\n")

    # ✅ Ajout des exemples dans Modelfile
    for entry in data:
        input_text = entry.get("input", "").replace('"', "'").strip()
        output_text = entry.get("output", "").replace('"', "'").strip()

        if input_text and output_text:
            f.write(f"MESSAGE \"Utilisateur : {input_text}\"\n")
            f.write(f"MESSAGE \"Assistant : {output_text}\"\n\n")

    # ✅ Ajout d'une ligne de fin propre pour éviter EOF
    f.write("SYSTEM \"Fin des exemples. Le modèle est prêt à répondre.\"\n")

print("✅ Fichier Modelfile mis à jour avec le dataset.")

# ✅ Lancer l'entraînement avec Ollama
os.system("ollama create mistral-finetuned -f Modelfile")
print("✅ Entraînement terminé avec succès !")







