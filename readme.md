# Pflegegutachten-Anamnese Optimierer

Dieses Python-Skript nutzt die OpenAI-API und Retrieval-Augmented Generation (RAG), um aus einem schnell erstellten, unstrukturierten Entwurf einer Anamnese eine klar strukturierte und professionelle Anamnese für die Pflegebegutachtung zu generieren. Der Fokus des Programms liegt auf der präzisen Verarbeitung der Pflegebegutachtungsrichtlinien und der detaillierten Formulierung des Prompts, der die Transformation der Anamnese ermöglicht.

## Funktionsweise

Das Skript verwendet einen zweistufigen Prozess:

1. **Vektordatenbank-Erstellung**:
   - Die Pflegebegutachtungsrichtlinien werden in einer FAISS-Vektordatenbank gespeichert
   - Der Text wird in kleine, überlappende Chunks aufgeteilt
   - Für jeden Chunk werden Embeddings erstellt

2. **Anamnese-Optimierung**:
   - Liest einen Rohtext einer Anamnese
   - Sucht relevante Passagen aus den Begutachtungsrichtlinien
   - Kombiniert diese mit einem systemischen Prompt
   - Generiert eine strukturierte und optimierte Version der Anamnese

### Kernfunktionalität

- **RAG-Integration**: Nutzt FAISS für effiziente Ähnlichkeitssuche in den Begutachtungsrichtlinien
- **Prompt-Engineering**: Der systemische Prompt ist so konzipiert, dass die KI die unstrukturierten Texte in eine klare, verständliche und strukturierte Form bringt
- **Textoptimierung**: Die Anamnese wird in ein professionelles und leicht verständliches Format transformiert, das Laien und Fachpersonen gleichermaßen anspricht

## Anforderungen

- **Python 3.7+**
- **Python-Bibliotheken**: 
  - `openai`
  - `python-dotenv`
  - `langchain`
  - `langchain-community`
  - `langchain-openai`
  - `faiss-cpu`
  - `pypdf`
  - `tiktoken`
  
Installation der Bibliotheken mit:
  
```bash
pip install -r requirements.txt
```

## Verzeichnisstruktur

```
.
├── rag_data/               # Enthält die Pflegebegutachtungsrichtlinien
│   └── pflege_richtlinien_adult.pdf
├── input/                  # Eingabetexte
│   └── maria_mustermann.txt
├── output/                 # Optimierte Anamnesen
│   └── optimized_document.txt
├── faiss_index/           # Gespeicherte Vektordatenbank
├── improver.py            # Hauptskript
└── requirements.txt       # Abhängigkeiten
```

## Erste Schritte

1. Erstellen Sie die notwendigen Verzeichnisse
2. Legen Sie die Pflegebegutachtungsrichtlinien als PDF in `rag_data/` ab
3. Platzieren Sie den zu optimierenden Text in `input/`
4. Führen Sie das Skript aus:
```bash
python improver.py
```

Die optimierte Anamnese wird im `output/` Verzeichnis gespeichert.
