import webbrowser
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.response.schema import StreamingResponse
import gradio as gr
import sys
import os
import openai
import logging
from llama_index.prompts import Prompt
import shutil
import sqlite3
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    SummaryIndex,
    LLMPredictor,
)
from llama_index.node_parser import (
    UnstructuredElementNodeParser,
    SentenceSplitter
)
import torch
from PIL import Image, ImageDraw
import faiss
import matplotlib.pyplot as plt
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index import ServiceContext
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import ChatMessage
from llama_index.llms import OpenAI
from llama_index.agent import OpenAIAgent, ContextRetrieverOpenAIAgent
from langchain.embeddings import OpenAIEmbeddings
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from torchvision import models, transforms
from transformers import AutoModelForObjectDetection
import fitz
import sys
sys.path.append('./llama_hub/tools/google_search/')
from base import GoogleSearchToolSpec
# Load environment variables from .env file
load_dotenv()
# _________________________________________________________________#
# Establish a connection to the SQLite database

conn = sqlite3.connect("chat_history.db")
cursor_1 = conn.cursor()
# Create a table to store chat history if it doesn't exist
cursor_1.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY,
        chat_history TEXT,
        source_inform TEXT,
        session_id INTEGER
    )
''')
cursor_2 = conn.cursor()
cursor_2.execute('''
    CREATE TABLE IF NOT EXISTS session_history (
        id INTEGER PRIMARY KEY,
        session_title TEXT
    )
''')

conn.commit()
conn.close()


# Adding the Theme here ##
wordlift_theme = gr.themes.Soft(
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

template = '''Abbiamo fornito le informazioni di contesto di seguito:{context_str}
Prendendo in considerazione queste informazioni, in qualità di Europlanner consapevole degli obiettivi e delle priorità dell'UE, ti preghiamo di fornire risposte e fonti alle seguenti domande.Devi sempre rispondere in italiano: {query_str}'''

custom_prompt = Prompt(template)
# Set the custom prompt


def set_prompt(prompt):
    global custom_prompt
    custom_prompt = Prompt(prompt)


tender_description = gr.State("")
company_description = gr.State("")
tender_description = "Questo è uno strumento che assiste nella redazione delle risposte relative al bando di gara ed ai relativi contenuti. Può essere impiegato per ottenere informazioni dettagliate sul bando, sul contesto normativo e sugli obiettivi dell'Unione Europea, dello Stato e della Regione. Utilizzalo per ottimizzare la tua strategia di risposta e per garantire la conformità con le linee guida e i requisiti specificati."
company_description = "Questo è uno strumento che assiste nella creazione di contenuti relativi all'azienda. Può essere utilizzato per rispondere a domande relative all'azienda."


def set_description(name, tool_description: str):
    if name == "tender_description":
        global tender_description
        tender_description = tool_description
    elif name == "company_description":
        global company_description
        company_description = tool_description


# _________________________________________________________________#
response_sources = ""
# Set model
model = gr.State('')
model = "gpt-4-1106-preview"
openai.api_key = os.getenv("openai_key")
service_context = gr.State('')
indices = {}
index_needs_update = {"company": True, "tender": True}
status = ""
source_infor_results = []
google_api_key = os.getenv('google_search_key')
google_engine_id = os.getenv('google_engine_id')
file_tender_inform_datas = []
file_company_inform_datas = []
current_session_id = ''
session_list = []
doc_ids = {"company": [], "tender": []}
company_doc_ids = []
tender_doc_ids = []
google_upload_url = ''
google_source_urls = [['No data', 'No data', 'No data', 'No data', 'No data',
                       'No data', 'No data', 'No data', 'No data', 'No data', 'No data']]

device = "cuda" if torch.cuda.is_available() else "cpu"

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# load table detection model
# processor = TableTransformerImageProcessor(max_size=800)
auto_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
).to(device)

# load table structure recognition model
# structure_processor = TableTransformerImageProcessor(max_size=1000)
structure_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to(device)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor(
        [width, height, width, height], dtype=torch.float32
    )
    return boxes


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [
        elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
    ]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


def detect_and_crop_save_table(
    file_path, cropped_table_directory="./data/table_images/"
):
    image = Image.open(file_path)

    filename, _ = os.path.splitext(file_path.split("/")[-1])

    if not os.path.exists(cropped_table_directory):
        os.makedirs(cropped_table_directory)

    # prepare image for the model
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = auto_model(pixel_values)

    # postprocess to get detected tables
    id2label = auto_model.config.id2label
    id2label[len(auto_model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    # print(f"number of tables detected {len(detected_tables)}")

    for idx in range(len(detected_tables)):
        #   # crop detected table out of image
        cropped_table = image.crop(detected_tables[idx]["bbox"])
        file_name = os.path.basename(filename)
        cropped_table.save(f"{cropped_table_directory}/{file_name}_{idx}.png")


def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

def set_chatting_mode(value):
    global chatting_mode_status
    chatting_mode_status = value


set_chatting_mode("Only Document")


def getSessionList():
    global current_session_id
    global session_list
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM session_history ORDER BY id DESC")

    rows = cursor.fetchall()

    session_list = []
    temp = []
    # Initialize chat history as an empty list
    # If there are entries, add them to the list
    if len(rows) > 0:
        # Get the last ID from the last row
        last_id = rows[0][0]

        if current_session_id == '':
            current_session_id = last_id
        # Assuming id is the first column
        for row in rows:
            # print(row[1])
            session_list.append(row[0])
            temp.append([row[1]])
    else:
        temp.append(['No Data'])
        gr.Info("You have to create Session")
    # Close the connection
    conn.commit()
    conn.close()
    return temp


getSessionList()
# _______________________________________________________
# pdf viewer


def pdf_view_url():
    # Use an HTML iframe element to embed the PDF viewer.
    pdf_viewer_html = f'<iframe src="file/assets/pdf_viewer.html" width="100%" height="470px"></iframe>'
    return pdf_viewer_html


def get_tender_files_inform(directory_path):
    global file_tender_inform_datas
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    files = os.listdir(directory_path)
    # file_names = [open(f'./data/tender/{file}','rb') for file in files]
    # print(f"---get_file---{file_names}----")
    if len(files) > 0:
        load_or_update_index('./data/tender/', 'tender')
    file_inform_data = []
    file_tender_inform_datas = []
    for file_number, file_name in enumerate(files, start=1):
        file_inform_data = [file_name, "Delete"]
        file_tender_inform_datas.append(file_inform_data)
    if file_tender_inform_datas:
        return file_tender_inform_datas
    else:
        return [['No File', ' ']]


def get_company_files_inform(directory_path):
    global file_company_inform_datas
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    files = os.listdir(directory_path)
    if len(files) > 0:
        load_or_update_index('./data/company/', 'company')
    file_inform_data = []
    file_company_inform_datas = []
    for file_number, file_name in enumerate(files, start=1):
        file_inform_data = [file_name, "Delete"]
        file_company_inform_datas.append(file_inform_data)
    if file_company_inform_datas:
        return file_company_inform_datas
    else:
        return [['No File', ' ']]

def extract_images_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        
        images = []
        
        # Create the 'data/image' folder if it doesn't exist
        output_folder = 'data/image'
        os.makedirs(output_folder, exist_ok=True)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            images_on_page = page.get_images(full=True)
            
            for img_index, img_info in enumerate(images_on_page):
                img_index = img_info[0]
                base_image = doc.extract_image(img_index)
                image_bytes = base_image["image"]
                
                # Save the image to the 'data/image' folder
                image_filename = f"image_page_{page_num + 1}_index_{img_index}.png"
                image_path = os.path.join(output_folder, image_filename)
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                images.append(image_path)
        
        return images
    except fitz.fitz.FileDataError as e:
        print(f"Error opening PDF '{pdf_path}': {e}")
        return []

def get_child_file_paths(directory_path):
    # Get a list of file names in the specified directory
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    return file_paths

# Function to load and preprocess an image
def load_and_preprocess_image(file_path):
    image = Image.open(file_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Function to extract features from an image using a pre-trained model
def extract_features(image_path, model):
    model.eval()
    img = load_and_preprocess_image(image_path)
    with torch.no_grad():
        features = model(img)
    return features.numpy()

# Function to create an index using Faiss
def create_image_index(image_folder, model):
    index = faiss.IndexFlatL2(2048)  # Assuming the model outputs 2048-dimensional features

    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]

    # Iterate over images and add them to the index
    for image_path in image_paths:
        features = extract_features(image_path, model)
        index.add(features)

    return index, image_paths

# ________________________________________________________
# Modified load_index function to handle multiple indices


def load_index(directory_path, index_key):
    global status
    global doc_ids
    documents = SimpleDirectoryReader(
        directory_path, filename_as_id=True).load_data()
        
    doc_ids[index_key] = [x.doc_id for x in documents]

    child_file_paths = get_child_file_paths(directory_path)
    for index, value in enumerate(child_file_paths):
        images = extract_images_from_pdf(value)    
        for file_path in images:
            detect_and_crop_save_table(file_path)

    # Load a pre-trained ResNet model
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.fc = torch.nn.Sequential()  # Remove the final fully connected layer

    image_index, image_paths = create_image_index('data/table_images', resnet_model)
    indices["image"] = image_index
    llm = OpenAI(temperature=0, model=model)
    service_context = ServiceContext.from_defaults(llm=llm)
    node_parser = SentenceSplitter()

    all_tools = []
    for index, value in enumerate(documents):
        summary = (
            f"This content about document. Use"
            f" this tool if you want to answer any questions about document.\n"
        )
        doc_tool = QueryEngineTool(
            query_engine=value,
            metadata=ToolMetadata(
                name=f"tool_{index}",
                description=summary,
            ),
        )
        all_tools.append(doc_tool)
    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)

    status += f"loaded documents with {len(documents)} pages.\n"
    if index_key in indices:
        # If index already exists, just update it
        status += f"Index for {index_key} loaded from memory.\n"
        logging.info(f"Index for {index_key} loaded from memory.")
        index = indices[index_key]
    else:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"./storage/{index_key}"
            )
            index = load_index_from_storage(storage_context)
            obj_index = ObjectIndex(index, tool_mapping)
            status += f"Index for {index_key} loaded from storage.\n"
            logging.info(f"Index for {index_key} loaded from storage.")
        except FileNotFoundError:
            # If index not found, create a new one
            logging.info(
                f"Index for {index_key} not found. Creating a new one...")
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(f"./storage/{index_key}")
            obj_index = ObjectIndex.from_objects(
                all_tools,
                tool_mapping,
                VectorStoreIndex,
                storage_context=storage_context
            )
            status += f"New index for {index_key} created and persisted to storage.\n"
            logging.info(
                f"New index for {index_key} created and persisted to storage.")
        nodes = node_parser.get_nodes_from_documents(documents)
        # build summary index
        summary_index = SummaryIndex(nodes, service_context=service_context)
        indices["summary"] = summary_index
        indices["top"] = obj_index
        # Save the loaded/created index in indices dict
        indices[index_key] = index
    index.refresh_ref_docs(documents)
    index.storage_context.persist(f"./storage/{index_key}")
    status += "Index refreshed and persisted to storage.\n"
    logging.info("Index refreshed and persisted to storage.")
    return index


def load_or_update_index(directory, index_key):
    global indices
    global index_needs_update
    global status
    if index_key not in index_needs_update or index_needs_update[index_key]:
        indices[index_key] = load_index(directory, index_key)
        index_needs_update[index_key] = False
    else:
        status += f"Index for {index_key} already up-to-date. No action taken.\n"
    return indices[index_key]

# Modified upload_file function to handle index_key


def upload_file(files, index_key):
    global index_needs_update
    global google_source_urls
    global google_upload_url
    gr.Info("Indexing(uploading...)Please check the Debug output")
    directory_path = f"data/{index_key}"
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    for file in files:
        # Define the destination file path
        destination_path = os.path.join(
            directory_path, os.path.basename(file.name))
        # Move the file from its temporary location to the directory_path
        shutil.move(file.name, destination_path)
    # Set index_needs_update for the specified index_key to True
    if index_key == "url":
        file_list = os.listdir(directory_path)
        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            # Check if the item in the directory is a file (not a subdirectory)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    # value = file_content.strip().split(',')

        # If splitting by comma resulted in only one value, split by white space
                    # if len(value) == 1:
                    # temp_arr=[]
                    google_source_urls = []
                    value = file_content.strip().split('\n')
                    value = value[:10]
                    for val in value:
                        google_upload_url += f"siteSearch={val}&"
                    if len(value) < 10:
                        value.extend(['No data'] * (10 - len(value)))
                    # temp_arr.append(['question']+value)
                    google_source_urls.append(['question']+value)
                    # Do something with the file content, e.g., print it
        # Load or update the index
    else:
        index_needs_update[index_key] = True
        load_or_update_index(directory_path, index_key)
        gr.Info("Documents are indexed")
    return "Files uploaded successfully!!!"


# _________________________________________________________________#
chat_history = []


def get_chat_history():
    # Create a new connection
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chat_history
        FROM chat_history
        WHERE session_id = ?
        ORDER BY id
    """, (current_session_id,))
    rows = cursor.fetchall()
    # print(f'---chat_history-----{rows}')
    # Initialize chat history as an empty list
    chat_history = []
    # If there are entries, add them to the list
    if rows:
        for row in rows:
            chat_history.append(list(row[0].split("::::")))
    # Close the connection
    conn.close()
    return chat_history


def write_chat_history_to_db(value, source_inform):
    # Create a new connection
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history(chat_history,source_inform,session_id) VALUES (?,?,?)",
                   (value, source_inform, current_session_id))

    conn.commit()
    conn.close()


def clear_chat_history():
    global current_session_id
    global chat_history
    global google_source_urls
    chat_history = []
    google_source_urls = [['No data', 'No data', 'No data', 'No data', 'No data',
                           'No data', 'No data', 'No data', 'No data', 'No data', 'No data']]

    # Clear the chat history from the database as well
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE session_id = ?",
                   (current_session_id,))
    cursor_1 = conn.cursor()
    cursor_1.execute("DELETE FROM session_history WHERE id = ?",
                     (current_session_id,))
    conn.commit()
    conn.close()
    current_session_id = ''
    directory_path = f"./temp/"
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    # Update the interface after clearing chat history
    return gr.update(value=get_chat_history())


async def bot(history, messages_history):
    # Get the chat history from the database
    # Define custom prompt
    if current_session_id == '':
        gr.Info("You have to create new session")
        return
    try:
        global response_sources
        global indices
        global google_source_urls
        global google_upload_url
        if openai.api_key == "":
            gr.Warning("Invalid OpenAI API key.")
            raise ValueError("Invalid OpenAI API key.")
        loaded_history = get_chat_history()
        history_message = []
        for history_data in loaded_history[-min(5, len(loaded_history)):]:
            history_message.append(ChatMessage(
                role="user", content=history_data[0]))
            history_message.append(ChatMessage(
                role="assistant", content=history_data[1]))

        try:
            company = indices.get("company")
            tender = indices.get("tender")
            summary = indices.get("summary")
            obj_index = indices.get("top")
            image_index = indices.get("image")
        except Exception as e:
            if str(e) == "expected string or bytes-like object":
                gr.Warning(
                    "Please enter a valid OpenAI API key or set the env key.")
                yield "Please enter a valid OpenAI API key or set the env key."
            yield "Index not found. Please upload the files first."
        tools = []
        message = history[-1][0]
        if chatting_mode_status == "Only Document":
            if tender is None and company is None:
                gr.Warning("Index not found. Please upload the files first.")
                yield "Index not found. Please upload the files first."

            elif tender is None:
                company_query_engine = company.as_query_engine(
                    similarity_top_k=5)
                summary_query_engine = summary.as_query_engine()
                tools = [
                    QueryEngineTool(
                    query_engine=company_query_engine,
                    metadata=ToolMetadata(
                        name='company_index',
                        description=f'{company_description}'
                    )),
                    QueryEngineTool(
                    query_engine=summary_query_engine,
                    metadata=ToolMetadata(
                        name="summary_tool",
                        description=(
                            "Useful for any requests that require a holistic summary"
                            f" of EVERYTHING. For questions about"
                            " more specific sections, please use the vector_tool."
                        ),
                    )),
                    QueryEngineTool(
                    query_engine=image_index,
                    metadata=ToolMetadata(
                        name="image_tool",
                        description=(
                            "Useful for any requests that require a image"
                        ),
                    ))]
            elif company is None:
                tender_query_engine = tender.as_query_engine(
                    similarity_top_k=5)
                summary_query_engine = summary.as_query_engine()
                tools = [QueryEngineTool(
                    query_engine=tender_query_engine,
                    metadata=ToolMetadata(
                        name='tender_index',
                        description=f'{tender_description}'
                    )),
                    QueryEngineTool(
                    query_engine=summary_query_engine,
                    metadata=ToolMetadata(
                        name="summary_tool",
                        description=(
                            "Useful for any requests that require a holistic summary"
                            f" of EVERYTHING. For questions about"
                            " more specific sections, please use the vector_tool."
                        ),
                    )),
                    QueryEngineTool(
                    query_engine=image_index,
                    metadata=ToolMetadata(
                        name="image_tool",
                        description=(
                            "Useful for any requests that require a image"
                        ),
                    ))]
            else:
                tender_query_engine = tender.as_query_engine(
                    similarity_top_k=5)
                company_query_engine = company.as_query_engine(
                    similarity_top_k=5)
                summary_query_engine = summary.as_query_engine()
                tools = [QueryEngineTool(
                    query_engine=tender_query_engine,
                    metadata=ToolMetadata(
                        name='tender_index',
                        description=f'{tender_description}'
                    )),
                    QueryEngineTool(
                    query_engine=company_query_engine,
                    metadata=ToolMetadata(
                        name='company_index',
                        description=f'{company_description}'
                    )),
                    QueryEngineTool(
                    query_engine=summary_query_engine,
                    metadata=ToolMetadata(
                        name="summary_tool",
                        description=(
                            "Useful for any requests that require a holistic summary"
                            f" of EVERYTHING. For questions about"
                            " more specific sections, please use the vector_tool."
                        ),
                    )),
                    QueryEngineTool(
                    query_engine=image_index,
                    metadata=ToolMetadata(
                        name="image_tool",
                        description=(
                            "Useful for any requests that require a image"
                        ),
                    ))]
            agent = OpenAIAgent.from_tools(
                tools, verbose=True, prompt=custom_prompt)
            if history_message:
                # qa_message=f"Devi rispondere in italiano."
                # history_message.append({"role": "user", "content": qa_message})
                agent.memory.set(history_message)
            qa_message = f"{message}.Devi rispondere in italiano."
            response = agent.stream_chat(qa_message)
            # content_list = [item.content for item in response.sources]
            # print(content_list)

            if response.sources:
                response_sources = response.source_nodes
                stream_token = ""
                for token in response.response_gen:
                    stream_token += token
                    yield history, messages_history
                if stream_token and message:
                    write_chat_history_to_db(
                        f"#{len(source_infor_results)}:{message}::::{stream_token}", get_source_info())
            else:
                history_message = []
                response_sources = "No sources found."
                qa_message = f"({message}).If parentheses content is saying hello,you have to say 'Ciao! Come posso aiutarti oggi?' but if not, you have to say 'mi spiace non ho trovato informazioni pertinenti.'.Devi rispondere in italiano. "
                history_message.append({"role": "user", "content": qa_message})
                content = openai_agent(history_message)

                partial_message = ""
                for chunk in content:
                    if chunk.choices[0].delta.content:
                        partial_message = partial_message + \
                            chunk.choices[0].delta.content
                        yield history, messages_history
                if partial_message and message:
                    write_chat_history_to_db(
                        f"{message}::::{partial_message}", "no_data")

        elif chatting_mode_status == "Documents and Search":
            if tender is None and company is None:
                gr.Warning("Index not found. Please upload the files first.")
                yield "Index not found. Please upload the files first."
            elif tender is None:
                company_query_engine = company.as_query_engine(
                    similarity_top_k=5)
                summary_query_engine = summary.as_query_engine()
                tools = [
                    QueryEngineTool(
                        query_engine=company_query_engine,
                        metadata=ToolMetadata(
                            name='company_index',
                            description=f'{company_description}'
                        )),
                    QueryEngineTool(
                        query_engine=summary_query_engine,
                        metadata=ToolMetadata(
                            name="summary_tool",
                            description=(
                                "Useful for any requests that require a holistic summary"
                                f" of EVERYTHING. For questions about"
                                " more specific sections, please use the vector_tool."
                            ),
                        )),
                    QueryEngineTool(
                    query_engine=image_index,
                    metadata=ToolMetadata(
                        name="image_tool",
                        description=(
                            "Useful for any requests that require a image"
                        ),
                    ))]
            elif company is None:
                tender_query_engine = tender.as_query_engine(
                    similarity_top_k=5)
                summary_query_engine = summary.as_query_engine()
                tools = [QueryEngineTool(
                    query_engine=tender_query_engine,
                    metadata=ToolMetadata(
                        name='tender_index',
                        description=f'{tender_description}'
                    )),
                    QueryEngineTool(
                    query_engine=summary_query_engine,
                    metadata=ToolMetadata(
                        name="summary_tool",
                        description=(
                            "Useful for any requests that require a holistic summary"
                            f" of EVERYTHING. For questions about"
                            " more specific sections, please use the vector_tool."
                        ),
                    )),
                    QueryEngineTool(
                    query_engine=image_index,
                    metadata=ToolMetadata(
                        name="image_tool",
                        description=(
                            "Useful for any requests that require a image"
                        ),
                    ))]
            else:
                tender_query_engine = tender.as_query_engine(
                    similarity_top_k=5)
                company_query_engine = company.as_query_engine(
                    similarity_top_k=5)
                summary_query_engine = summary.as_query_engine()
                tools = [QueryEngineTool(
                    query_engine=tender_query_engine,
                    metadata=ToolMetadata(
                        name='tender_index',
                        description=f'{tender_description}'
                    )),
                    QueryEngineTool(
                    query_engine=company_query_engine,
                    metadata=ToolMetadata(
                        name='company_index',
                        description=f'{company_description}'
                    )),
                    QueryEngineTool(
                    query_engine=summary_query_engine,
                    metadata=ToolMetadata(
                        name="summary_tool",
                        description=(
                            "Useful for any requests that require a holistic summary"
                            f" of EVERYTHING. For questions about"
                            " more specific sections, please use the vector_tool."
                        ),
                    )),
                    QueryEngineTool(
                    query_engine=image_index,
                    metadata=ToolMetadata(
                        name="image_tool",
                        description=(
                            "Useful for any requests that require a image"
                        ),
                    ))]
            google_spec = GoogleSearchToolSpec(
                key=google_api_key, engine=google_engine_id, siteSearch=google_upload_url)
            google_tools = LoadAndSearchToolSpec.from_defaults(
                google_spec.to_tool_list()[0]
            ).to_tool_list()
            agent = OpenAIAgent.from_tools(
                [*tools, *google_tools], tool_retriever=obj_index.as_retriever(similarity_top_k=3), verbose=True, prompt=custom_prompt)
            if history_message:
                # qa_message=f"Devi rispondere in italiano."
                # history_message.append({"role": "user", "content": qa_message})
                agent.memory.set(history_message)
            qa_message = f"{message}.Devi rispondere in italiano."
            response = agent.stream_chat(qa_message)
            source_urls = google_spec.get_source_url(qa_message)
            stream_token = ""
            if response.source_nodes == []:
                temp_arry = []
                temp_arry.append(message)
                for source_url in source_urls:
                    temp_arry.append(source_url['link'])
                if google_source_urls[0][0] == 'No data':
                    google_source_urls = []
                    google_source_urls.append(temp_arry)
                else:
                    google_source_urls.append(temp_arry)
                # print(google_source_urls)

            elif response.source_nodes:
                response_sources = response.source_nodes
            else:
                response_sources = "No sources found."

            for token in response.response_gen:
                stream_token += token
                yield history, messages_history

            if stream_token and message:
                write_chat_history_to_db(
                    f"#{len(source_infor_results)}:{message}::::{stream_token}", get_source_info())
        else:
            history_message = []
            for history_data in loaded_history[-min(5, len(loaded_history)):]:
                history_message.append(
                    {"role": "user", "content": history_data[0]})
                history_message.append(
                    {"role": "assistant", "content": history_data[1]})

            history_message.append({"role": "user", "content": message})
            qa_message = f"Devi rispondere in italiano."
            history_message.append({"role": "user", "content": qa_message})
            content = openai_agent(history_message)

            partial_message = ""
            for chunk in content:
                if chunk.choices[0].delta.content:
                    partial_message = partial_message + \
                        chunk.choices[0].delta.content
                    yield history, messages_history
            if partial_message and message:
                write_chat_history_to_db(
                    f"{message}::::{partial_message}", "no_data")

    except ValueError as e:
        # Display the warning message in the Gradio interface
        gr.Warning(str(e))


def update_debug_info(upload_file):
    debug_info = status
    return debug_info


def update_source_info():
    global response_sources
    source_info = "Sources: \n"
    if response_sources == "No sources found.":
        return source_info
    else:
        for node_with_score in response_sources:
            # Exract the Node object
            node = node_with_score.node
            text = node.text
            # Extract file_name and page_label from extra_info
            extra_info = node.extra_info or {}
            file_name = extra_info.get("file_name")
            page_label = extra_info.get("page_label")
            # Append extracted information to debug_info string
            source_info += f"File Name: {file_name}\n, Page Label: {page_label}\n\n"

        return source_info


def update_source():
    global response_sources
    global source_infor_results
    source_infor_results = []
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
        FROM chat_history
        WHERE session_id = ?
        ORDER BY id
    """, (current_session_id,))
    rows = cursor.fetchall()
    # Initialize chat history as an empty list
    source_informs = []
    # If there are entries, add them to the list
    if rows:
        for row in rows:
            if row[2] != "no_data":
                temp = list(row[2].split("&&&&"))
                temp.append(list(row[1].split("::::"))[0])
                source_informs.append(temp)
    # Close the connection
    if source_informs:
        for source_inform in source_informs:
            for temp in source_inform[:-2]:
                temp_1 = list(temp.split("::::"))
                temp_1.append(source_inform[-1])
                source_infor_results.append(temp_1)
        conn.close()
        # return chat_history
        final_result = []
        for result in source_infor_results:
            temp = []
            temp.append(result[-1])
            temp.append(result[0])
            temp.append(result[1])
            final_result.append(temp)
        if final_result:
            return final_result
        else:
            return [['No Data', 'No Data', 'No Data']]
    else:
        return [['No Data', 'No Data', 'No Data']]


def get_source_info():
    global response_sources
    source_info = ""
    if response_sources == "No sources found.":
        return source_info
    else:
        for node_with_score in response_sources:
            # Extract the Node object
            node = node_with_score.node
            text = node.text
            # Extract file_name and page_label from extra_info
            extra_info = node.extra_info or {}
            file_name = extra_info.get("file_name")
            page_label = extra_info.get("page_label")
            # Append extracted information to debug_info string
            source_info += f"{file_name}::::{page_label}::::{text}&&&&"
        return source_info


def delete_index(index_key):
    if openai.api_key:
        gr.Info("Deleting index..")
        global status
        directory_path = f"./storage/{index_key}"
        backup_path = f"./backup_path/{index_key}"
        documents_path = f"data/{index_key}"
        # remove the directory and its contents
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        if not os.path.exists(documents_path):
            os.makedirs(documents_path)
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        for filename in os.listdir(documents_path):
            source_file = os.path.join(documents_path, filename)
            destination_file = os.path.join(backup_path, filename)
            shutil.move(source_file, destination_file)
        shutil.rmtree(directory_path)
        shutil.rmtree(documents_path)

        for item in doc_ids[index_key]:
            indices[index_key].delete_ref_doc(item, delete_from_docstore=True)

        status = ""
        gr.Info("Index is deleted")
        debug_info = status
        return debug_info
    else:
        return None


def delete_row(index_key, file_name):
    global doc_ids
    current_datetime = datetime.now().timestamp()
    if openai.api_key:
        gr.Info("Deleting index..")
        backup_path = f"./backup_path/{index_key}/{current_datetime}"
        index_path = f"./storage/{index_key}"
        documents_path = f"./data/{index_key}/{file_name}"
        directory_path = f"data/{index_key}"
        if not os.path.exists(documents_path):
            os.makedirs(documents_path)
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        if not os.path.exists(index_path):
            os.makedirs(index_path)
        shutil.move(documents_path, backup_path)
        shutil.rmtree(index_path)

        string_to_match = f'data\\{index_key}\\{file_name}'
        filtered_list = [item for item in doc_ids[index_key]
                         if item.startswith(string_to_match)]
        for item in filtered_list:
            indices[index_key].delete_ref_doc(item, delete_from_docstore=True)
        # shutil.rmtree(documents_path)
        # index_needs_update[index_key] = True
        # indices[index_key].refresh_ref_docs()
        id = indices[index_key].index_id
        # print(os.path.abspath(documents_path))
        # print(indices[index_key].docstore.doc_ids)
        # load_or_update_index(directory_path, index_key)
        status = ""
        gr.Info("Index is deleted")
        debug_info = status
        return debug_info
    else:
        return None
# Set model


def set_model(_model):
    global model
    model = _model


def get_sources(choice):
    if choice == "view source":
        return gr.Textbox.update(value=response_sources)
# add state


def set_openai_api_key(api_key: str):
    if api_key:
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        llm_predictor = LLMPredictor(llm=ChatOpenAI(
            temperature=0, model_name=model, streaming=True))
        global service_context
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, chunk_size=1024)
    else:
        gr.Warning("Please enter a valid OpenAI API key or set the env key.")


def openai_agent(prompt):
    response = openai.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=1.0,
        stream=True
    )
    return response


def update_company_info(upload_file):

    file_value = get_company_files_inform(directory_path=f"data/company")
    return gr.update(value=file_value)


def update_tender_info(upload_file):

    file_value = get_tender_files_inform(directory_path=f"data/tender")
    return gr.update(value=file_value)


def set_tender_pdf(evt: gr.SelectData):
    select_data = evt.index
    if select_data[1] == 0:
        pdf_viewer_content = f'<iframe src="file/data/tender/{evt.value}" width="100%" height="600px"></iframe>'
        file_path = search_files_by_name("./data", evt.value)
        webbrowser.open(file_path)
        return gr.update(value=pdf_viewer_content)
    else:
        delete_row("tender", file_tender_inform_datas[int(select_data[0])][0])


def set_company_pdf(evt: gr.SelectData):
    select_data = evt.index
    if select_data[1] == 0:
        pdf_viewer_content = f'<iframe src="file/data/company/{evt.value}" width="100%" height="600px"></iframe>'
        file_path = search_files_by_name("./data", evt.value)
        webbrowser.open(file_path)
        return gr.update(value=pdf_viewer_content)
    else:
        delete_row(
            "company", file_company_inform_datas[int(select_data[0])][0])


def set_source_pdf(evt: gr.SelectData):
    pdf_viewer_content = f'<iframe src="file/data/company/{evt.value}" width="100%" height="800px"></iframe>'
    return gr.update(value=pdf_viewer_content)


def set_highlight_pdf(evt: gr.SelectData):
    select_data = evt.index
    file_name = source_infor_results[int(select_data[0])][0]
    source_text = source_infor_results[int(select_data[0])][2]
    file_path = search_files_by_name("./data", file_name)
    webbrowser.open(file_path)
    pdf_viewer_content = f'<h4>{source_text}</h4>'
    return gr.update(value=pdf_viewer_content)


def search_files_by_name(root_dir, file_name):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == file_name:
                return os.path.join(foldername, filename)

    # return found_files


def update_session(update_data):
    old_session = getSessionList()
    new_session = update_data.values

    for i in range(min(len(new_session), len(old_session))):
        nested_list1 = new_session[i]
        nested_list2 = old_session[i]

        # Convert the nested lists to strings for comparison
        str1 = str(nested_list1)
        str2 = str(nested_list2)

        # Check if the nested lists are different
        if str1 != str2:
            conn = sqlite3.connect("chat_history.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE session_history SET session_title = ? WHERE id = ?;",
                           (nested_list1[0], session_list[i]))
            conn.commit()
            conn.close()


def add_session(session_title):
    if session_title:
        global current_session_id
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO session_history(session_title) VALUES (?)", (session_title,))
        current_session_id = cursor.lastrowid
        # print(f'late id :{last_inserted_id}')
        current_session_id
        conn.commit()
        conn.close()
        return gr.update(value=getSessionList())
    else:
        gr.Info("You have to input Sesstion title")
        return gr.update(value=getSessionList())


def set_session(evt: gr.SelectData):
    global current_session_id
    select_data = evt.index
    current_session_id = session_list[int(select_data[0])]
# _________________________________________________________________#
# Define the Gradio interface

# Load custom CSS
with open(
    "./assets/custom.css",
    "r",
    encoding="utf-8",
) as f:
    customCSS = f.read()

# Add the title
title = f"""<h2 align="center">BLM Mostro <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACiklEQVR4AWIYSoAP0Hw5wMoRhVE4i8zOvtq2bSNOjbC2jaC27Tasbdt2g9q27duT5Ezy93a9M807ybd392LO9fwLmoIF4AJ4BX6Bb+AROAAmgOput9trp3F2MA98AkpgGb8BSuMO6O5yucyYXTmK/uALUBytIo+0UfZkvs4NUD16d5crJT73AMWRD8ZoUiPdxLwNQKqsqFuRpidEx/tG7E2jC2z8DOQVZQlIjoH+2myZNK9r5Xk8HjeSWUCRcRGYuw0kh0SjLzBNHqCD+YGuikDXJKAE3UFITQBKoymIWj6fz83NqAQ/uFwBVZLr9QIUBW3BNrAMxKLS3IQTaDqFnbiA5Ql4TLewQmttfY1onblUXp/PdIvfJpJb9Gis18/PghvsnVNqTZ9TesFQFrzjZrJdmAEDyQqgSF5ZfkLufFDfTnOepH36iZA33hd5y4E1aJTSxj60BefAN+GzyCrMyoxzMMV358SN2J5+R+TxU1yf/6Hi9LuSaDqQnRlnMEUZnXTmndL2ryWAqb4J74FVNm/C1uCE5rNEVjilHcOGwDZxMAeAEvSUdUaJi6iqgydgHVCkoCwvzMxrfI87fRVfME3zH59dLGweIDSLWvo7RXsZtQ7UpryIggqyIxvAkjhex1e4vCVFrHHJFWJQs+wKSEzTHygg+QUqh9soZ2TorYdk3NF5A8+gJgYhgv4grDKCK2I5cmodPKQ/iBfMB1DTyvN6vW4k04T5LL8/IeINnp7Rr+KDB3Lk64KE5aVF3fKgstWejDAMI2JzGSGPAT8i+GPSnfl6vQeclbhUECwTHbH4xGv7mTQlzzhrSeM115elM5fhhhZcfGDAMQ/UZfjlrNwQlBQXjhnPc/4Aqf7wDR6odCYAAAAASUVORK5CYII=" alt="client" style="display: inline;"></h2>"""

clear = gr.Button("🧹 Start fresh")

with gr.Blocks(css=customCSS, theme=wordlift_theme) as demo:
    gr.Info("Please enter a valid OpenAI API key or set the env key.")
    gr.HTML(value=title)

    session_state = gr.State([])
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            session_title = gr.Textbox(label="Session Title")
            new_session_btn = gr.Button(value="New Session")
            session_list_dataframe = gr.Dataframe(value=getSessionList,
                                                  headers=["Session List"],
                                                  datatype=["str"],
                                                  col_count=(1, "fixed"),
                                                  height=600,
                                                  interactive=True,
                                                  elem_id="session_dataframe")
        with gr.Column(scale=15):
            with gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(value=get_chat_history(),
                        elem_id="chuanhu_chatbot", height="850px")

                    msg = gr.Textbox(
                        label=" 🪄",
                        placeholder="Type a message to the bot and press enter",
                        container=False,
                    )
                with gr.Column(scale=4):
                    source_dataframe = gr.Dataframe(value=update_source,
                                                    headers=[
                                                        "question", "File name", "page numeber"],
                                                    datatype=[
                                                        "str", "str", "str"],
                                                    col_count=(3, "fixed"),
                                                    row_count=5,
                                                    wrap=False,
                                                    height=300,
                                                    label="Agent Mode Q&A history",
                                                    interactive=False,
                                                    elem_id="source_dataframe"
                                                    )
                    pdf_viewer_html = gr.HTML(
                        value=pdf_view_url, label="preview", elem_id="pdf_reference")
                    clear = gr.Button("🧹 Start fresh")
            with gr.Accordion("⚙️ Settings", open=False):
                with gr.Tab("history"):
                    google_search_dataframe = gr.Dataframe(
                        value=google_source_urls,
                        headers=["Question", "Source URL-1", "Source URL-2", "Source URL-3", "Source URL-4",
                                 "Source URL-5", "Source URL-6", "Source URL-7", "Source URL-8", "Source URL-9", "Source URL-10"],
                        datatype=["str", "str", "str", "str", "str",
                                  "str", "str", "str", "str", "str", "str"],
                        col_count=(11, "fixed"),
                        label="Google search source url",
                        interactive=False
                    )

                openai_api_key_textbox = gr.Textbox(
                    placeholder="Paste your OpenAI API key (sk-...)",
                    show_label=False,
                    lines=1,
                    type="password",
                )
                openai_api_key_textbox.change(
                    set_openai_api_key, inputs=openai_api_key_textbox)
                # chatting_mode_slide=gr.Slider(1,3,
                #                             step=1,
                #                             label="Chatting Quality > Agent Type",
                #                             container=True ,
                #                             info="Only Document > No Documents > Documents and Search",
                #                             interactive=True)
                chatting_mode_radio = gr.Radio(
                    value="Only Document", choices=["Only Document", "No Documents", "Documents and Search"], label="Chatting Quality > Agent Type"
                )
                # gr.HTML(value=f"<div style='display:inline'><span style='position: absolute;left: 0px;'>Low</span><span style='position: absolute;right: 0px;'>High</span></div>")
                custom_prompt = gr.Textbox(
                    placeholder="Here goes the custom prompt",
                    value=template,
                    lines=5,
                    label="Custom Prompt",
                )

                custom_prompt.change(fn=set_prompt, inputs=custom_prompt)

                tender_description_textbox = gr.Textbox(
                    placeholder="Here goes the tender description",
                    value=tender_description,
                    lines=5,
                    label="Tender Description",
                )

                tender_description_textbox.change(lambda x: set_description(
                    "tender_description", x), inputs=tender_description_textbox)

                company_description_textbox = gr.Textbox(
                    placeholder="Here goes the company description",
                    value=company_description,
                    lines=5,
                    label="Company Description",
                )

                company_description_textbox.change(lambda x: set_description(
                    "company_description", x), inputs=company_description_textbox)

                radio = gr.Radio(
                    value="gpt-4-1106-preview", choices=["gpt-3.5-turbo", "gpt-4-1106-preview"], label="Models"
                )

                radio.change(set_model, inputs=radio)

                with gr.Row():
                    tender_data = get_tender_files_inform(
                        directory_path=f"data/tender")
                    company_data = get_company_files_inform(
                        directory_path=f"data/company")

                    tender_dataframe = gr.Dataframe(value=tender_data,
                                                    headers=[
                                                        "Tender File Name", "Action"],
                                                    datatype=["str", "str"],
                                                    col_count=(2, "fixed"),
                                                    label="Tender File list",
                                                    interactive=False
                                                    )

                    company_dataframe = gr.Dataframe(value=company_data,
                                                     headers=[
                                                         "Company File Name", "Action"],
                                                     datatype=["str", "str"],
                                                     col_count=(2, "fixed"),
                                                     label="Company File list",
                                                     interactive=False
                                                     )
                with gr.Tab("📁 Add your files here"):
                    with gr.TabItem("Tender documents"):

                        upload_button1 = gr.UploadButton(
                            file_types=[".pdf", ".csv", ".docx", ".txt"], file_count="multiple"
                        )
                    with gr.TabItem("Company files"):
                        upload_button2 = gr.UploadButton(
                            file_types=[".pdf", ".csv", ".docx", ".txt"], file_count="multiple"
                        )
                with gr.Tab("❌ Delete ALL Indices"):
                    with gr.TabItem("Tender documents"):
                        delete_button1 = gr.Button(
                            value="❌ Delete Tender Index"
                        )
                    with gr.TabItem("Company files"):
                        delete_button2 = gr.Button(
                            value="❌ Delete Company Index"
                        )
                with gr.Tab("🔍 Add your own google search url"):
                    with gr.TabItem("google search url"):
                        upload_button3 = gr.UploadButton(
                            file_types=[".txt"], file_count="multiple"
                        )

            with gr.Accordion("🔍 Context", open=False, visible=False):
                sources = gr.Textbox(
                    placeholder="Sources will be shown here",
                    lines=5,
                    label="Sources",
                )
            # Debug Accordion
            with gr.Accordion("🔍 Debug", open=False):
                debug_output = gr.Textbox(
                    placeholder="Debug output will be printed here",
                    lines=5,
                    label="Debug Output",
                )

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, session_state], [chatbot, session_state]).then(
        lambda: gr.update(value=get_chat_history()), None, outputs=[chatbot]).then(
        lambda: gr.update(value=update_source()), None, outputs=source_dataframe).then(
            update_source_info, None, outputs=sources
    ).then(
        lambda: gr.update(value=google_source_urls), None, outputs=google_search_dataframe)

    file_response1 = upload_button1.upload(
        lambda files: upload_file(files, "tender"), upload_button1
    ).then(
        update_tender_info, inputs=[tender_dataframe], outputs=tender_dataframe
    )
    file_response2 = upload_button2.upload(
        lambda files: upload_file(files, "company"), upload_button2
    ).then(
        update_company_info, inputs=[
            company_dataframe], outputs=company_dataframe
    )
    file_response3 = upload_button3.upload(
        lambda files: upload_file(files, "url"), upload_button3
    ).then(
        lambda: gr.update(value=google_source_urls), None, outputs=google_search_dataframe)
    file_response1.then(
        update_debug_info, inputs=[upload_button1], outputs=debug_output
    )
    file_response2.then(
        update_debug_info, inputs=[upload_button2], outputs=debug_output
    )

    delete_button1.click(lambda: delete_index("tender")).then(
        update_tender_info, inputs=[tender_dataframe], outputs=tender_dataframe
    )
    delete_button2.click(lambda: delete_index("company")).then(
        update_company_info, inputs=[
            company_dataframe], outputs=company_dataframe
    )

    chatting_mode_radio.change(set_chatting_mode, inputs=[chatting_mode_radio])

    clear.click(lambda: [], None, chatbot, queue=False).then(
        clear_chat_history, None, outputs=[chatbot]).then(
        lambda: gr.update(value=update_source()), None, outputs=source_dataframe).then(
        lambda: gr.update(value=getSessionList()), None, outputs=session_list_dataframe).then(
        lambda: gr.update(value=google_source_urls), None, outputs=google_search_dataframe)

    tender_dataframe.select(set_tender_pdf, None, pdf_viewer_html).then(
        update_tender_info, inputs=[tender_dataframe], outputs=tender_dataframe
    )
    company_dataframe.select(set_company_pdf, None, pdf_viewer_html).then(
        update_company_info, inputs=[
            company_dataframe], outputs=company_dataframe
    )
    source_dataframe.select(set_highlight_pdf, None, pdf_viewer_html)

    session_list_dataframe.select(set_session, None, None).then(
        lambda: gr.update(value=get_chat_history()), None, outputs=[chatbot]).then(
        lambda: gr.update(value=update_source()), None, outputs=source_dataframe).then(
            update_source_info, None, outputs=sources).then(
        lambda: gr.update(value=google_source_urls), None, outputs=google_search_dataframe)

    session_list_dataframe.input(update_session, inputs=[
                                 session_list_dataframe])

    new_session_btn.click(add_session, inputs=[session_title], outputs=[
                          session_list_dataframe])

demo.queue().launch(inline=True).then(lambda: gr.update(
    value=get_chat_history()), None, outputs=[chatbot])
