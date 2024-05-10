# config.py
import os
import gradio as gr
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration Constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_ENGINE_ID = os.getenv("GOOGLE_ENGINE_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WORDLIFT_THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#007AFF",
        c100="rgba(0, 122, 255, 0.2)",
        c200="#007AFF",
        c300="rgba(0, 122, 255, 0.32)",
        c400="rgba(0, 122, 255, 0.32)",
        c500="rgba(0, 122, 255, 1.0)",
        c600="rgba(0, 122, 255, 1.0)",
        c700="rgba(0, 122, 255, 0.32)",
        c800="rgba(0, 122, 255, 0.32)",
        c900="#007AFF",
        c950="#007AFF",
    ),
    secondary_hue=gr.themes.Color(
        c50="#576b95",
        c100="#576b95",
        c200="#576b95",
        c300="#576b95",
        c400="#576b95",
        c500="#576b95",
        c600="#576b95",
        c700="#576b95",
        c800="#576b95",
        c900="#576b95",
        c950="#576b95",
    ),
    neutral_hue=gr.themes.Color(
        name="gray",
        c50="#f9fafb",
        c100="#f3f4f6",
        c200="#e5e7eb",
        c300="#d1d5db",
        c400="#B2B2B2",
        c500="#808080",
        c600="#636363",
        c700="#515151",
        c800="#393939",
        c900="#272727",
        c950="#171717",
    ),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    button_primary_background_fill="#2196F3",
    button_primary_background_fill_dark="#2196F3",
    button_primary_background_fill_hover="#007AFF",
    button_primary_border_color="#2196F3",
    button_primary_border_color_dark="#2196F3",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#F2F2F2",
    button_secondary_background_fill_dark="#2B2B2B",
    button_secondary_text_color="#393939",
    button_secondary_text_color_dark="#FFFFFF",
    block_title_text_color="*primary_500",
    block_title_background_fill="*primary_100",
    input_background_fill="#F6F6F6",
)
TEMPLATE = '''
Abbiamo fornito le informazioni di contesto di seguito: {context_str}

Prendendo in considerazione queste informazioni e utilizzando le risorse online disponibili, ti preghiamo di effettuare una ricerca approfondita sull'argomento. Cerca dati, statistiche, studi, articoli, e qualsiasi altra fonte rilevante che possa arricchire la risposta.

In qualità di Europlanner consapevole degli obiettivi e delle priorità dell'UE, ti preghiamo di fornire una sintesi delle informazioni trovate, citando le fonti. Assicurati di valutare la credibilità e l'attualità delle informazioni raccolte.

Devi sempre rispondere in italiano e fornire risposte dettagliate e ben argomentate alle seguenti domande: {query_str}

Riassumi i punti chiave e le scoperte principali, includendo eventuali collegamenti diretti alle fonti online quando possibile.
'''
DATABASE_PATH = "../chat_history.db"