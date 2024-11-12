# Pflegegutachten-Anamnese Optimierer

Dieses Python-Skript nutzt die OpenAI-API, um aus einem schnell erstellten, unstrukturierten Entwurf einer Anamnese eine klar strukturierte und professionelle Anamnese für die Pflegebegutachtung zu generieren. Der Fokus des Programms liegt nicht auf der eigentlichen Programmierung, sondern auf der präzisen und detaillierten Formulierung des Prompts, der die Transformation der Anamnese ermöglicht. Das Ergebnis ist eine standardisierte Anamnese zur Ermittlung des Pflegegrades.

## Funktionsweise

Das Skript liest einen Rohtext einer Anamnese, übermittelt diesen zusammen mit einem systemischen Prompt an ein KI-Modell von OpenAI und gibt eine strukturierte und sprachlich optimierte Version der Anamnese zurück. 

### Kernfunktionalität

- **Prompt-Engineering**: Der systemische Prompt ist so konzipiert, dass die KI die unstrukturierten Texte in eine klare, verständliche und strukturierte Form bringt.
- **Textoptimierung**: Die Anamnese wird in ein professionelles und leicht verständliches Format transformiert, das Laien und Fachpersonen gleichermaßen anspricht.

## Anforderungen

- **Python 3.7+**
- **Python-Bibliotheken**: `openai`, `python-dotenv`
  
  Installation der Bibliotheken mit:
  
  ```bash
  pip install openai python-dotenv
