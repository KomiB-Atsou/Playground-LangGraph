import os
import re
import pandas as pd
from typing import List, TypedDict, Dict, Any, Optional
import time
from bs4 import BeautifulSoup
import cloudscraper
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from urllib.parse import urljoin

# --- 1. Configuration initiale ---
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Clé API GOOGLE_API_KEY non trouvée. Veuillez la définir dans le fichier .env")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- 2. Définition des schémas de données ---

class AnalyseAnnonce(BaseModel):
    """Schéma de données pour l'analyse d'une annonce immobilière."""
    prix: Optional[int] = Field(description="Le prix du logement en euros (loyer mensuel ou prix de vente). Mettre None si non trouvé.")
    prix_m2: Optional[int] = Field(description="Le prix au mètre carré en euros. Mettre None si non trouvé.")
    surface: Optional[int] = Field(description="La surface en mètres carrés (m²). Mettre None si non trouvé.")
    nombre_pieces: Optional[int] = Field(description="Le nombre de pièces. Mettre None si non trouvé.")
    ville: Optional[str] = Field(description="La ville où se situe le logement. Mettre None si non trouvé.")
    description_courte: str = Field(description="Un résumé très court de l'annonce (1 phrase).")
    url: str = Field(description="L'URL de l'annonce analysée.")

class AgentState(TypedDict):
    """État du graphe qui circule entre les agents."""
    listing_page_url: str
    criteres_filtrage: Dict[str, Any]
    urls_a_traiter: List[str]
    annonces_valides: List[Dict[str, Any]]
    session: Any # Utiliser Any pour la compatibilité avec cloudscraper

# --- 3. Définition des Agents (nœuds du graphe) ---

def explorer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent qui explore la page de listing pour trouver les URLs des annonces individuelles.
    """
    print("--- AGENT: Exploration de la page de listing ---")
    url = state['listing_page_url']
    print(f"URL de la page: {url}")

    try:
        session = state['session']
        response = session.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # NOTE: Cette logique est à adapter pour chaque site cible.
        # On cherche tous les liens <a> qui semblent pointer vers une annonce.
        # Ici, on se base sur des mots-clés courants dans les URLs d'annonces.
        liens_annonces = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Regex adaptée pour bienici.com
            # On cherche des URLs qui commencent par /annonce/vente/
            if href.startswith('/annonce/vente/'):
                # On reconstruit l'URL absolue si elle est relative
                full_url = urljoin(url, href)
                liens_annonces.add(full_url)

        urls_trouvees = list(liens_annonces)
        print(f"-> {len(urls_trouvees)} URLs d'annonces trouvées.")
        return {"urls_a_traiter": urls_trouvees, "annonces_valides": []}

    except Exception as e:
        print(f"Erreur lors de l'exploration de {url}: {e}")
        return {"urls_a_traiter": [], "annonces_valides": []}

def analyzer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent qui traite une par une les URLs, les analyse, les filtre et les ajoute aux résultats valides.
    """
    print("\n--- AGENT: Analyse et filtrage des annonces ---")
    urls_a_traiter = state['urls_a_traiter']
    criteres = state['criteres_filtrage']
    annonces_valides = []
    session = state['session']

    structured_llm = llm.with_structured_output(AnalyseAnnonce)

    for i, url in enumerate(urls_a_traiter):
        print(f"\n[{i+1}/{len(urls_a_traiter)}] Traitement de : {url}")
        time.sleep(2) # Ajout d'un délai de 2 secondes pour être plus discret

        # 1. Scraping
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            contenu_texte = ' '.join(soup.get_text(separator=' ', strip=True).split())
            if not contenu_texte:
                print("  -> Contenu vide, annonce ignorée.")
                continue
        except Exception as e:
            print(f"  -> Erreur de scraping: {e}, annonce ignorée.")
            continue

        # 2. Analyse avec Gemini
        prompt = f"""
        Analyse le contenu de l'annonce immobilière suivante et extrais les informations demandées.
        Le contenu provient de l'URL : {url}.

        Contenu de l'annonce :
        ---
        {contenu_texte[:8000]}
        ---
        """
        try:
            analyse = structured_llm.invoke(prompt)
            analyse.url = url # On s'assure que l'URL est bien présente
        except Exception as e:
            print(f"  -> Erreur d'analyse LLM: {e}, annonce ignorée.")
            continue

        # 3. Filtrage
        respecte_criteres = True
        if analyse.prix is not None and 'prix_vente_max' in criteres and analyse.prix > criteres['prix_vente_max']:
            respecte_criteres = False
        if analyse.prix_m2 is not None and 'prix_m2_max' in criteres and analyse.prix_m2 > criteres['prix_m2_max']:
            respecte_criteres = False
        if analyse.surface is not None and 'surface_min' in criteres and analyse.surface < criteres['surface_min']:
            respecte_criteres = False
        if analyse.nombre_pieces is not None and 'pieces_min' in criteres and analyse.nombre_pieces < criteres['pieces_min']:
            respecte_criteres = False

        if respecte_criteres:
            print("  -> CRITÈRES RESPECTÉS. Annonce ajoutée.")
            annonces_valides.append(analyse.dict())
        else:
            print("  -> Critères non respectés, annonce ignorée.")

    return {"annonces_valides": annonces_valides}

def excel_writer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent final qui écrit les annonces validées dans un fichier Excel.
    """
    print("\n--- AGENT: Génération du fichier Excel ---")
    annonces = state['annonces_valides']
    if not annonces:
        print("Aucune annonce valide à écrire. Le fichier ne sera pas créé.")
        return {}

    df = pd.DataFrame(annonces)
    # Réorganiser les colonnes pour une meilleure lisibilité
    colonnes_ordonnees = ['prix', 'prix_m2', 'surface', 'nombre_pieces', 'ville', 'description_courte', 'url']
    df = df[[col for col in colonnes_ordonnees if col in df.columns]]

    nom_fichier = "resultats_annonces.xlsx"
    df.to_excel(nom_fichier, index=False)
    print(f"-> Fichier '{nom_fichier}' créé avec succès avec {len(df)} annonces.")
    return {}

# --- 4. Construction et exécution du graphe ---

# Définition du graphe
workflow = StateGraph(AgentState)

# Ajout des noeuds (agents)
workflow.add_node("explorer", explorer_agent)
workflow.add_node("analyzer", analyzer_agent)
workflow.add_node("excel_writer", excel_writer_agent)

# Définition des transitions (flux de travail)
workflow.set_entry_point("explorer")
workflow.add_edge("explorer", "analyzer")
workflow.add_edge("analyzer", "excel_writer")
workflow.add_edge("excel_writer", END)

# Compilation du graphe
app = workflow.compile()

# --- 5. Lancement du système ---

if __name__ == "__main__":
    # --- À CONFIGURER AVANT DE LANCER ---
    # L'URL de la page contenant la liste des annonces
    url_de_listing = "https://www.bienici.com/recherche/vente/montpellier-34000?prix-max=500000&surface-min=50&nb-pieces-min=3"

    # Les critères pour filtrer les annonces
    criteres = {
        "prix_vente_max": 500000,
        "prix_m2_max": 4500,
        "surface_min": 50,
        "pieces_min": 3
    }
    # --- FIN DE LA CONFIGURATION ---
    
    # Création d'une session avec cloudscraper pour contourner les protections anti-bot
    session = cloudscraper.create_scraper()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/125.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })

    # Définition de l'état initial
    inputs = {
        "session": session,
        "listing_page_url": url_de_listing,
        "criteres_filtrage": criteres
    }

    # Lancement du graphe
    print("--- DÉBUT DU TRAITEMENT ---")
    result = app.invoke(inputs)

    print("\n--- FIN DU TRAITEMENT ---")
    print("Résultats finaux :")
    # Affiche un résumé des annonces trouvées
    for annonce in result.get('annonces_valides', []):
        print(f"- {annonce.get('prix')}€ ({annonce.get('prix_m2')}€/m²), {annonce.get('surface')}m², {annonce.get('ville')}: {annonce.get('url')}")
