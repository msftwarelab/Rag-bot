import gradio as gr
import sys, os
import openai
import logging
from llama_index.prompts import Prompt
import shutil
import sqlite3
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    LLMPredictor,
)
from llama_index import ServiceContext
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import ChatMessage 
from llama_index.agent import OpenAIAgent,ContextRetrieverOpenAIAgent
from langchain.embeddings import OpenAIEmbeddings
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
import sys
sys.path.append('./llama_hub/tools/google_search/')
from base import GoogleSearchToolSpec
from llama_index.response.schema import StreamingResponse
from llama_index.query_engine import SubQuestionQueryEngine
import pandas as pd
import fitz
import time
import json
from dotenv import load_dotenv
import webbrowser
# Load environment variables from .env file
load_dotenv()
# _________________________________________________________________#
# Establish a connection to the SQLite database
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()
# Create a table to store chat history if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY,
        chat_history TEXT,
        source_inform TEXT
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
Prendendo in considerazione queste informazioni, in qualità di Europlanner consapevole degli obiettivi e delle priorità dell'UE, ti preghiamo di fornire risposte e fonti alle seguenti domande: {query_str}'''

custom_prompt = Prompt(template)
# Set the custom prompt
def set_prompt(prompt):
    global custom_prompt
    custom_prompt = Prompt(prompt)
    
tender_description = gr.State("")
company_description = gr.State("")
tender_description = "Questo è uno strumento che assiste nella redazione delle risposte relative al bando di gara ed ai relativi contenuti. Può essere impiegato per ottenere informazioni dettagliate sul bando, sul contesto normativo e sugli obiettivi dell'Unione Europea, dello Stato e della Regione. Utilizzalo per ottimizzare la tua strategia di risposta e per garantire la conformità con le linee guida e i requisiti specificati."
company_description = "Questo è uno strumento che assiste nella creazione di contenuti relativi all'azienda. Può essere utilizzato per rispondere a domande relative all'azienda."

def set_description(name, tool_description : str):
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
model = "gpt-3.5-turbo"
openai.api_key = os.getenv("openai_key")
service_context = gr.State('')
indices = {}
index_needs_update = {"company": True, "tender": True}
status = ""
source_infor_results=[]
google_api_key = os.getenv('google_search_key')
google_engine_id=os.getenv('google_engine_id')

       
def set_chatting_mode(value):
    global chatting_mode_status
    chatting_mode_status=value

set_chatting_mode(1)
#_______________________________________________________ 
#pdf viewer
def pdf_view_url():
    # Use an HTML iframe element to embed the PDF viewer.
    pdf_viewer_html = f'<iframe src="file/assets/pdf_viewer.html" width="100%" height="550px"></iframe>'
    return pdf_viewer_html

def get_files_inform(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    files=os.listdir(directory_path)
    file_inform_data=[]
    file_inform_datas=[]
    for file_number,file_name in enumerate(files,start=1):
        file_inform_data=[file_name]
        file_inform_datas.append(file_inform_data)
    if file_inform_datas:
        return file_inform_datas
    else:
        return [['No File']]
#________________________________________________________
# Modified load_index function to handle multiple indices
def load_index(directory_path, index_key):
    global status
    documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
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
            status += f"Index for {index_key} loaded from storage.\n"
            logging.info(f"Index for {index_key} loaded from storage.")
        except FileNotFoundError:
            # If index not found, create a new one
            logging.info(f"Index for {index_key} not found. Creating a new one...")
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(f"./storage/{index_key}")
            status += f"New index for {index_key} created and persisted to storage.\n"
            logging.info(f"New index for {index_key} created and persisted to storage.")
        # Save the loaded/created index in indices dict
        indices[index_key] = index
    refreshed_docs = index.refresh_ref_docs(
        documents, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
    )
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
    gr.Info("Indexing(uploading...)Please check the Debug output")
    directory_path = f"data/{index_key}"
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    for file in files:
        # Define the destination file path
        destination_path = os.path.join(directory_path, os.path.basename(file.name))
        # Move the file from its temporary location to the directory_path
        shutil.move(file.name, destination_path)
    # Set index_needs_update for the specified index_key to True
    index_needs_update[index_key] = True
    # Load or update the index
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
        ORDER BY id
    """)
    rows = cursor.fetchall()
    # Initialize chat history as an empty list
    chat_history = []
    # If there are entries, add them to the list
    if rows:
        for row in rows:
            chat_history.append(list(row[0].split("::::")))
    # Close the connection
    conn.close()
    return chat_history

def write_chat_history_to_db(value,source_inform):
    # Create a new connection
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history(chat_history,source_inform) VALUES (?,?)", (value,source_inform))
    conn.commit()
    conn.close()
    
def clear_chat_history():
    global chat_history
    chat_history = []
    # Clear the chat history from the database as well
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    directory_path = f"./temp/"
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    return gr.update(value=get_chat_history())  # Update the interface after clearing chat history

async def bot(message,history):
    # Get the chat history from the database
    # Define custom prompt
    try:
        global response_sources
        global indices
        if openai.api_key == "":
            gr.Warning("Invalid OpenAI API key.")
            raise ValueError("Invalid OpenAI API key.")
        loaded_history = get_chat_history()
        history_message=[]
        for history_data in loaded_history[-min(5, len(loaded_history)):]:
            history_message.append(ChatMessage(role="user", content=history_data[0]))
            history_message.append(ChatMessage(role="assistant", content=history_data[1]))
       
        try:
            company = indices.get("company")
            tender = indices.get("tender")
        except Exception as e:
            if str(e) == "expected string or bytes-like object":
                gr.Warning("Please enter a valid OpenAI API key or set the env key.")
                yield "Please enter a valid OpenAI API key or set the env key."
            yield "Index not found. Please upload the files first."
        
        if chatting_mode_status==1:
            if tender is None and company is None:
                gr.Warning("Index not found. Please upload the files first.")
                yield "Index not found. Please upload the files first."
            
            elif tender is None:
                company_query_engine = company.as_query_engine(similarity_top_k=5)
                tools = [
                    QueryEngineTool(
                    query_engine=company_query_engine,
                    metadata=ToolMetadata(
                        name='company_index',
                        description=f'{company_description}'
                    ))]
            elif company is None:
                tender_query_engine = tender.as_query_engine(similarity_top_k=5)
                tools = [QueryEngineTool(
                    query_engine=tender_query_engine,
                    metadata=ToolMetadata(
                        name='tender_index',
                        description=f'{tender_description}'
                    ))]
            else:
                tender_query_engine = tender.as_query_engine(similarity_top_k=5)
                company_query_engine = company.as_query_engine(similarity_top_k=5)
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
                    ))]
            agent = OpenAIAgent.from_tools(tools, verbose=True, prompt=custom_prompt)
            # if history_message:
            #     agent.memory.set(history_message)
            response = agent.stream_chat(message)
            # content_list = [item.content for item in response.sources]
            # print(content_list)
            
            if response.sources:
                response_sources = response.source_nodes
                stream_token=""
                for token in response.response_gen:
                    stream_token += token
                    yield stream_token
                if stream_token and message:
                    write_chat_history_to_db(f"#{len(source_infor_results)}:{message}::::{stream_token}",get_source_info())
            else:
                history_message=[]
                response_sources = "No sources found."
                qa_message=f"({message}).If parentheses content is saying hello,you have to say 'Hello! How can I assist you today?' but if not, you have to say 'Sorry,I don' t know it'. "
                history_message.append({"role": "user", "content": qa_message})
                content = openai_agent(history_message)
                
                partial_message=""
                for chunk in content:
                    if len(chunk['choices'][0]['delta']) != 0:
                        partial_message = partial_message + chunk['choices'][0]['delta']['content']
                        yield partial_message
                if partial_message and message:
                    write_chat_history_to_db(f"{message}::::{partial_message}","no_data")
                #  for token in agent.aquery(message):
                #     stream_token += token
                #     yield stream_token
                
            
        
        elif chatting_mode_status==3:
            if tender is None and company is None:
                gr.Warning("Index not found. Please upload the files first.")
                yield "Index not found. Please upload the files first."
            elif tender is None:
                company_query_engine = company.as_query_engine(similarity_top_k=5)
                tools = [
                    QueryEngineTool(
                    query_engine=company_query_engine,
                    metadata=ToolMetadata(
                        name='company_index',
                        description=f'{company_description}'
                    ))]
            elif company is None:
                tender_query_engine = tender.as_query_engine(similarity_top_k=5)
                tools = [QueryEngineTool(
                    query_engine=tender_query_engine,
                    metadata=ToolMetadata(
                        name='tender_index',
                        description=f'{tender_description}'
                    ))]
            else:
                tender_query_engine = tender.as_query_engine(similarity_top_k=5)
                company_query_engine = company.as_query_engine(similarity_top_k=5)
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
                    ))]
            google_spec = GoogleSearchToolSpec(key=google_api_key, engine=google_engine_id)
            google_tools = LoadAndSearchToolSpec.from_defaults(
            google_spec.to_tool_list()[0]
            ).to_tool_list()    
            agent = OpenAIAgent.from_tools([*tools,*google_tools], verbose=True, prompt=custom_prompt)
            if history_message:
                agent.memory.set(history_message)
            response = agent.stream_chat(message)
            stream_token=""
            if response.sources:
                response_sources = response.source_nodes
            else:
                response_sources = "No sources found."
                
            for token in response.response_gen:
                stream_token += token
                yield stream_token

            if stream_token and message:
                write_chat_history_to_db(f"#{len(source_infor_results)}:{message}::::{stream_token}",get_source_info())    
        else:
            history_message=[]
            for history_data in loaded_history[-min(5, len(loaded_history)):]:
                history_message.append({"role": "user", "content": history_data[0] })
                history_message.append({"role": "assistant", "content":history_data[1]})
                
            history_message.append({"role": "user", "content": message})
            content = openai_agent(history_message)

            partial_message=""
            for chunk in content:
                if len(chunk['choices'][0]['delta']) != 0:
                    partial_message = partial_message + chunk['choices'][0]['delta']['content']
                    yield partial_message
            if partial_message and message:
                write_chat_history_to_db(f"{message}::::{partial_message}","no_data")
                    
    except ValueError as e:
        gr.Warning(str(e))  # Display the warning message in the Gradio interface
def update_debug_info(upload_file):
    debug_info = status
    return debug_info
def update_source_info(chatbot):
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
    source_infor_results=[]
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chat_history,source_inform
        FROM chat_history
        ORDER BY id
    """)
    rows = cursor.fetchall()
    # Initialize chat history as an empty list
    source_informs = []
    # If there are entries, add them to the list
    if rows:
        for row in rows:
            if row[1] != "no_data":
                temp=list(row[1].split("&&&&"))
                temp.append(list(row[0].split("::::"))[0])
                source_informs.append(temp)
    # Close the connection
    if source_informs:
        for source_inform in source_informs:
            for temp in source_inform[:-2]:
                temp_1=list(temp.split("::::"))
                temp_1.append(source_inform[-1])
                source_infor_results.append(temp_1)
        conn.close()
        # return chat_history
        final_result=[]
        for result in source_infor_results:
            temp=[]
            temp.append(result[-1])
            temp.append(result[0])
            temp.append(result[1])
            final_result.append(temp)
        if final_result:
            return final_result
        else:
            return [['No Data','No Data','No Data']]
    else:
        return [['No Data','No Data','No Data']]
    
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
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=model, streaming=True))
        global service_context
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=1024)
    else:
        gr.Warning("Please enter a valid OpenAI API key or set the env key.")
        

def openai_agent(prompt):
    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
        temperature=1.0,
        stream=True
    )
    return response

def update_company_info(upload_file):
    
    file_value=get_files_inform(directory_path = f"data/company")
    return gr.update(value=file_value)

def update_tender_info(upload_file):
    
    file_value=get_files_inform(directory_path = f"data/tender")
    return gr.update(value=file_value)

def set_tender_pdf(evt: gr.SelectData):
    # pdf_viewer_content = f'<iframe src="file/data/tender/{evt.value}" width="100%" height="600px"></iframe>'
    file_path=search_files_by_name("./data",evt.value)
    file_url=f"https://wordlift-ai-content-writer.hf.space/file{file_path}"
    webbrowser.open(file_url.replace("./", "/"))
    # return gr.update(value=pdf_viewer_content)
def set_company_pdf(evt: gr.SelectData):
    # pdf_viewer_content = f'<iframe src="file/data/company/{evt.value}" width="100%" height="600px"></iframe>'
    file_path=search_files_by_name("./data",evt.value)
    file_url=f"https://wordlift-ai-content-writer.hf.space/file{file_path}"
    webbrowser.open(file_url.replace("./", "/"))
    # return gr.update(value=pdf_viewer_content)
def set_source_pdf(evt: gr.SelectData):
    pdf_viewer_content = f'<iframe src="file/data/company/{evt.value}" width="100%" height="800px"></iframe>'
    return gr.update(value=pdf_viewer_content)

def set_highlight_pdf(evt: gr.SelectData):
    select_data=evt.index
    file_name=source_infor_results[int(select_data[0])][0]
    source_text=source_infor_results[int(select_data[0])][2]
    file_path=search_files_by_name("./data",file_name)
    file_url=f"https://wordlift-ai-content-writer.hf.space/file{file_path}"
    webbrowser.open(file_url.replace("./", "/"))
    # print(source_infor_results)
    # pdf_viewer_content = f'<iframe src="file/{file_path}" width="100%" height="600px"></iframe>'
    pdf_viewer_content = f'<h4>{source_text}</h4>'
    return gr.update(value=pdf_viewer_content)

def search_files_by_name(root_dir, file_name):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == file_name:
                return os.path.join(foldername, filename)

    # return found_files
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
title = """<h2 align="center">BLM Mostro <img src="file/assets/client.png" alt="client" style="display: inline;"></h2>"""

chatbot = gr.Chatbot(value=get_chat_history(),elem_id="chuanhu_chatbot", height="850px")

msg = gr.Textbox(
        label=" 🪄",
        placeholder="Type a message to the bot and press enter",
        container=False,
    )
clear = gr.Button("🧹 Start fresh")

with gr.Blocks(css=customCSS, theme=wordlift_theme) as demo:
    gr.Info("Please enter a valid OpenAI API key or set the env key.")
    gr.HTML(value=title)
    with gr.Row():
        with gr.Column(scale=6):
            chat_interface=gr.ChatInterface(
                                bot, 
                                chatbot=chatbot,
                                textbox=msg,
                                retry_btn=None,
                                undo_btn=None,
                                clear_btn=clear,
                                submit_btn=None
                            )
            chat_interface
        with gr.Column(scale=4):
            source_dataframe=gr.Dataframe(value=update_source,
                                        headers=["question","File name","page numeber"],
                                        datatype=["str","str","str"],
                                        col_count=(3,"fixed"),
                                        max_rows=5,
                                        overflow_row_behaviour="show_ends",
                                        height=300,
                                        label="Agent Mode Q&A history",
                                        interactive=False,
                                        elem_id="source_dataframe"
                                        )
            pdf_viewer_html=gr.HTML(value=pdf_view_url,label="preview",elem_id="pdf_reference")
    with gr.Accordion("⚙️ Settings", open=False):
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )
        openai_api_key_textbox.change(set_openai_api_key, inputs=openai_api_key_textbox)
        chatting_mode_slide=gr.Slider(1,3,
                                      step=1,
                                      label="Chatting Quality > Agent Type",
                                      container=True ,
                                      info="Only Document > No Documents > Documents and Search",
                                      interactive=True)
        # gr.HTML(value=f"<div style='display:inline'><span style='position: absolute;left: 0px;'>Low</span><span style='position: absolute;right: 0px;'>High</span></div>")
        custom_prompt = gr.Textbox(
            placeholder="Here goes the custom prompt",
            value= template,
            lines=5,
            label="Custom Prompt",
        )
        
        custom_prompt.change(fn=set_prompt, inputs=custom_prompt)
        
        tender_description_textbox = gr.Textbox(
            placeholder="Here goes the tender description",
            value= tender_description,
            lines=5,
            label="Tender Description",
        )
        
        tender_description_textbox.change(lambda x: set_description("tender_description",x), inputs=tender_description_textbox)
        
        company_description_textbox = gr.Textbox(
            placeholder="Here goes the company description",
            value= company_description,
            lines=5,
            label="Company Description",
        )
        
        company_description_textbox.change(lambda x: set_description("company_description",x), inputs=company_description_textbox)
        
        radio = gr.Radio(
            value="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"], label="Models"
        )
        
        radio.change(set_model, inputs=radio)
        
        with gr.Row():
            tender_data=get_files_inform(directory_path = f"data/tender")
            company_data=get_files_inform(directory_path = f"data/company")
            tender_dataframe=gr.Dataframe(value=tender_data,
                                        headers=["Tender File Name"],
                                        datatype=["str"],
                                        col_count=(1,"fixed"),
                                        label="Tender File list",
                                        interactive=False
                                        )
            company_dataframe=gr.Dataframe(value=company_data,
                                        headers=["Company File Name"],
                                        datatype=["str"],
                                        col_count=(1,"fixed"),
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
    
    with gr.Accordion("🔍 Context", open=False):
        sources =  gr.Textbox(
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
    
    response = msg.submit(update_source_info, inputs=[chatbot], outputs=sources).then(
        lambda:gr.update(value=get_chat_history()),None, outputs=[chatbot]).then(
        lambda:gr.update(value=update_source()),None,outputs=source_dataframe).then(
            update_source_info, inputs=[chatbot], outputs=sources
        )

    file_response1 = upload_button1.upload(
        lambda files: upload_file(files,"tender"), upload_button1
    ).then(
        update_tender_info, inputs=[tender_dataframe], outputs=tender_dataframe
    )
    file_response2 = upload_button2.upload(
        lambda files: upload_file(files,"company"), upload_button2
    ).then(
        update_company_info, inputs=[company_dataframe], outputs=company_dataframe
    )
    
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
        update_company_info, inputs=[company_dataframe], outputs=company_dataframe
    )
    chatting_mode_slide.change(set_chatting_mode,inputs=[chatting_mode_slide])
    clear.click(lambda: [], None, chatbot, queue=False).then(
        clear_chat_history,None,outputs=[chatbot]).then(
        lambda:gr.update(value=update_source()),None,outputs=source_dataframe)
    
    tender_dataframe.select(set_tender_pdf,None,pdf_viewer_html)
    company_dataframe.select(set_company_pdf,None,pdf_viewer_html)
    source_dataframe.select(set_highlight_pdf,None,pdf_viewer_html)

demo.queue().launch(inline=True).then(lambda:gr.update(value=get_chat_history()),None, outputs=[chatbot])