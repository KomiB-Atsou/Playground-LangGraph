import os
import cloudscraper
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional

# --- Configuration ---
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Clé API GOOGLE_API_KEY non trouvée. Veuillez la définir dans le fichier .env")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- Schéma de données pour l'analyse ---
class AnalyseAnnonce(BaseModel):
    """Schéma de données pour l'analyse d'une annonce immobilière."""
    prix: Optional[int] = Field(description="Le prix du logement en euros. Mettre None si non trouvé.")
    surface: Optional[int] = Field(description="La surface en mètres carrés (m²). Mettre None si non trouvé.")
    nombre_pieces: Optional[int] = Field(description="Le nombre de pièces. Mettre None si non trouvé.")
    ville: Optional[str] = Field(description="La ville où se situe le logement. Mettre None si non trouvé.")
    description_courte: str = Field(description="Un résumé très court de l'annonce (1 phrase).")
    url: str = Field(description="L'URL de l'annonce analysée.")

def analyze_url(url_to_analyze: str):
    """
    Scrape une URL unique, l'analyse avec Gemini et affiche le résultat.
    """
    print(f"--- Tentative d'analyse de l'URL : {url_to_analyze} ---")

    # 1. Scraping avec cloudscraper
    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url_to_analyze, timeout=15)
        response.raise_for_status()
        print("-> Succès : La page a été téléchargée.")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        contenu_texte = ' '.join(soup.get_text(separator=' ', strip=True).split())
        
        if not contenu_texte:
            print("-> Erreur : Le contenu de la page est vide.")
            return

    except Exception as e:
        print(f"-> Erreur lors du scraping : {e}")
        print("Le site a probablement bloqué la requête.")
        return

    # 2. Analyse avec Gemini
    print("\n--- Analyse du contenu avec Gemini ---")
    prompt = f"""
    Analyse le contenu de l'annonce immobilière suivante et extrais les informations demandées.
    Le contenu provient de l'URL : {url_to_analyze}.

    Contenu de l'annonce :
    ---
    {contenu_texte[:8000]}
    ---
    """
    try:
        structured_llm = llm.with_structured_output(AnalyseAnnonce)
        analyse = structured_llm.invoke(prompt)
        analyse.url = url_to_analyze
        
        print("\n--- RÉSULTAT DE L'ANALYSE ---")
        print(f"  Prix: {analyse.prix} €")
        print(f"  Surface: {analyse.surface} m²")
        print(f"  Pièces: {analyse.nombre_pieces}")
        print(f"  Ville: {analyse.ville}")
        print(f"  Description: {analyse.description_courte}")
        print(f"  URL: {analyse.url}")

    except Exception as e:
        print(f"-> Erreur lors de l'analyse par le LLM : {e}")

if __name__ == "__main__":
    # --- URL de l'annonce à analyser ---
    # Essayez avec une annonce de Le Bon Coin (échouera probablement)
    # url = "https://www.leboncoin.fr/ventes_immobilieres/2633391295.htm"
    
    # Essayez avec une annonce de Bien'ici (devrait fonctionner)
    url = "https://www.bienici.com/annonce/vente/montpellier-34000/appartement/5-pieces/401m2/p-80320493"
    
    analyze_url(url)
