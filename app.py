import webbrowser
from dotenv import load_dotenv
from datetime import datetime
import gradio as gr
import os
import json
import openai
import logging
import shutil
import sqlite3
from llama_index.core.prompts import PromptTemplate
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.legacy.llm_predictor.base import LLMPredictor
from llama_index.legacy.service_context import ServiceContext
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext
from langchain_community.chat_models import ChatOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.llms.llm import ChatMessage
from llama_index.agent.openai import OpenAIAgent
from llama_hub.tools.google_search.base import GoogleSearchToolSpec
from llama_hub.tools.tavily_research import TavilyToolSpec

from llama_hub.llama_packs.ragatouille_retriever.base import RAGatouilleRetrieverPack
# from llama_index.core.llama_pack import download_llama_pack
from src.config import DATABASE_PATH, WORDLIFT_THEME, GOOGLE_API_KEY, GOOGLE_ENGINE_ID, TAVILY_API_KEY, OPENAI_API_KEY, TEMPLATE
from src.database import create_tables, add_chat_history, get_chat_history, delete_chat_history
from src.utilities import pdf_view_url, get_available_storage, check_or_create_directory

load_dotenv()

# RAGatouilleRetrieverPack = download_llama_pack(
#     "RAGatouilleRetrieverPack", "./ragatouille_pack"
# )

openai.api_key = OPENAI_API_KEY

class RagBot:
    def __init__(self):
        create_tables()
        self.custom_prompt = PromptTemplate(TEMPLATE)
        self.tender_description = gr.State("")
        self.company_description = gr.State("")
        self.tender_description = "Questo è uno strumento che assiste nella redazione delle risposte relative al bando di gara ed ai relativi contenuti. Può essere impiegato per ottenere informazioni dettagliate sul bando, sul contesto normativo e sugli obiettivi dell'Unione Europea, dello Stato e della Regione. Utilizzalo per ottimizzare la tua strategia di risposta e per garantire la conformità con le linee guida e i requisiti specificati."
        self.company_description = "Questo è uno strumento che assiste nella creazione di contenuti relativi all'azienda. Può essere utilizzato per rispondere a domande relative all'azienda."
        self.response_sources = ""
        self.model = gr.State('')
        self.model = "gpt-4-turbo-preview"
        self.colbert = gr.State('')
        self.colbert = "No"
        self.service_context = gr.State('')
        self.indices = {}
        self.index_needs_update = {"company": True, "tender": True}
        self.status = ""
        self.source_infor_results = []
        self.file_tender_inform_datas = []
        self.file_company_inform_datas = []
        self.current_session_id = 0
        self.session_list = []
        self.doc_ids = {"company": [], "tender": []}
        self.documents = []
        self.ragatouille_pack = gr.State('')
        self.company_doc_ids = []
        self.tender_doc_ids = []
        self.google_upload_url = ''
        self.google_source_urls = [['No data', 'No data', 'No data', 'No data', 'No data', 'No data', 'No data', 'No data', 'No data', 'No data', 'No data']]
        self.chatting_mode_status = ""
        self.chat_history = []
        
        self.set_chatting_mode("Documents and Search")
        self.set_model(self.model)
        self.getSessionList()

    def set_prompt(self, prompt):
        self.custom_prompt = PromptTemplate(prompt)

    def set_description(self, name, tool_description: str):
        if name == "tender_description":
            self.tender_description = tool_description
        elif name == "company_description":
            self.company_description = tool_description

    def set_chatting_mode(self, value):
        self.chatting_mode_status = value

    def set_model(self, _model):
        if _model == 'gpt-4-turbo-preview':
            _model = 'gpt-4-0125-preview'
        self.model = _model

    def set_colbert(self, _colbert):
        self.colbert = _colbert
        self.initRAGatouille()

    def getSessionList(self):
        temp = []
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM session_history ORDER BY id DESC")
                rows = cursor.fetchall()
                self.session_list = []
                if rows:
                    self.session_list = [row[0] for row in rows]
                    temp = [[row[1]] for row in rows]
                    if not self.current_session_id:
                        self.current_session_id = rows[0][0]
                else:
                    temp = [["No Data"]]
                    self.current_session_id = 0
                    gr.Info("You have to create Session")
        except sqlite3.Error as e: 
            temp = [["No Data"]]
            gr.Info("You have to create Session")
        return temp

    def get_tender_files_inform(self, directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)

            files = os.listdir(directory_path)

            if files:
                self.load_or_update_index(f"data/tender/{self.current_session_id}/", 'tender')

            self.file_tender_inform_datas = [[file_name, "Delete"] for file_name in files]

            return self.file_tender_inform_datas or [['No File', ' ']]

        except OSError as e:
            gr.Info(f"File system error: {str(e)}")
            return [['Error', 'File System Issue']]

        except Exception as e:
            gr.Info(f"Unexpected error: {str(e)}")
            return [['Error', 'Unexpected Issue']]


    def get_company_files_inform(self, directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)

            files = os.listdir(directory_path)

            if files:
                self.load_or_update_index(f"data/company/{self.current_session_id}/", 'company')

            self.file_company_inform_datas = [[file_name, "Delete"] for file_name in files]

            return self.file_company_inform_datas or [['No File', ' ']]

        except OSError as e:
            gr.Info(f"File system error: {str(e)}")
            return [['Error', 'File System Issue']]

        except Exception as e:
            gr.Info(f"Unexpected error: {str(e)}")
            return [['Error', 'Unexpected Issue']]

    def load_index(self, directory_path, index_key):
        documents = SimpleDirectoryReader(
            directory_path, filename_as_id=True).load_data()
        self.doc_ids[index_key] = [x.doc_id for x in documents]
        # print(documents.id_)
        self.status += f"loaded documents with {len(documents)} pages.\n"
        if index_key in self.indices:
            # If index already exists, just update it
            self.status += f"Index for {index_key} loaded from memory.\n"
            logging.info(f"Index for {index_key} loaded from memory.")
            index = self.indices[index_key]
        else:
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=f"./storage/{index_key}"
                )
                index = load_index_from_storage(storage_context)
                self.status += f"Index for {index_key} loaded from storage.\n"
                logging.info(f"Index for {index_key} loaded from storage.")
            except FileNotFoundError:
                # If index not found, create a new one
                logging.info(
                    f"Index for {index_key} not found. Creating a new one...")
                index = VectorStoreIndex.from_documents(documents)
                index.storage_context.persist(f"./storage/{index_key}")
                self.status += f"New index for {index_key} created and persisted to storage.\n"
                logging.info(
                    f"New index for {index_key} created and persisted to storage.")
            # Save the loaded/created index in indices dict
            self.indices[index_key] = index
        index.refresh_ref_docs(documents)
        index.storage_context.persist(f"./storage/{index_key}")
        self.status += "Index refreshed and persisted to storage.\n"
        logging.info("Index refreshed and persisted to storage.")
        return index



    def load_or_update_index(self, directory, index_key):
        print(get_available_storage())
        if index_key not in self.index_needs_update or self.index_needs_update[index_key]:
            self.indices[index_key] = self.load_index(directory, index_key)
            self.index_needs_update[index_key] = False
        else:
            self.status += f"Index for {index_key} already up-to-date. No action taken.\n"
        return self.indices[index_key]

    def initRAGatouille(self):
        self.documents = []

        directory_paths = [f"./data/tender/{self.current_session_id}", f"./data/company/{self.current_session_id}"]

        for directory_path in directory_paths:
            if os.path.isdir(directory_path):
                try:
                    docs = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
                    self.documents.extend(docs)
                except Exception as e:
                    print(f"Error processing directory {directory_path}: {e}")
                    raise
            else:
                print(f"Warning: Directory '{directory_path}' not found")

    def upload_file(self, files, index_key):
        gr.Info("Indexing(uploading...)Please check the Debug output")
        directory_path = f"data/{index_key}/{self.current_session_id}"
        check_or_create_directory(directory_path)
        company = self.indices.get("company")

        for file in files:
            destination_path = os.path.join(directory_path, os.path.basename(file.name))
            shutil.move(file.name, destination_path)

        if index_key == "url":
            file_list = os.listdir(directory_path)
            for file_name in file_list:
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        file_content = file.read()
                        self.google_source_urls = []
                        value = file_content.strip().split('\n')
                        value = value[:10]
                        for val in value:
                            self.google_upload_url += f"siteSearch={val}&"
                        if len(value) < 10:
                            value.extend(['No data'] * (10 - len(value)))
                        self.google_source_urls.append(['question']+value)
            print(self.google_source_urls)
            google_spec = GoogleSearchToolSpec(key=GOOGLE_API_KEY, engine=GOOGLE_ENGINE_ID, num=1)
            if self.google_source_urls[0][0] != 'No data':
                for search_url in self.google_source_urls[0][1:]:
                    if search_url == 'No data':
                        break
                    search_results = google_spec.google_search(search_url)

                    for result in search_results:
                        result_dict = json.loads(result.text)
                        snippet = result_dict['items'][0]['snippet']
                        node = Document(text=snippet)
                        company.insert_nodes([node])
                    print("The Google search result has been successfully inserted.")
        else:
            self.index_needs_update[index_key] = True
            self.load_or_update_index(directory_path, index_key)
            gr.Info("Documents are indexed")
        return "Files uploaded successfully!!!"

    def clear_chat_history(self):
        self.chat_history = []
        self.google_soucre_urls = [['No data', 'No data', 'No data', 'No data', 'No data','No data', 'No data', 'No data', 'No data', 'No data', 'No data']]
        delete_chat_history(self.current_session_id)
        self.current_session_id = ''
        directory_path = f"./temp/"
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        return gr.update(value=get_chat_history(self.current_session_id))


    async def bot(self, history, messages_history):
        if not self.current_session_id:
            gr.Info("You have to create a new session")
            return
        try:
            if not openai.api_key:
                gr.Warning("Invalid OpenAI API key.")
                raise ValueError("Invalid OpenAI API key.")

            loaded_history = get_chat_history(self.current_session_id)
            history_message = self.build_history_message(loaded_history)

            try:
                company = self.indices.get("company")
                tender = self.indices.get("tender")
            except KeyError as e:
                gr.Warning("Please enter a valid OpenAI API key or set the env key.")
                yield history, messages_history
                return

            if not (tender or company):
                gr.Warning("Index not found. Please upload the files first.")
                yield history, messages_history
                return

            message = history[-1][0]

            if self.chatting_mode_status == "Only Document":
                tools = self.build_tools(company, tender)
                agent = OpenAIAgent.from_tools(tools, verbose=True, prompt=self.custom_prompt)
                if history_message:
                    agent.memory.set(history_message)

                response = await self.handle_only_document_mode(agent, message)

            elif self.chatting_mode_status == "Documents and Search":
                tools = self.build_tools(company, tender) + TavilyToolSpec(api_key=TAVILY_API_KEY).to_tool_list()
                agent = OpenAIAgent.from_tools(tools, verbose=True, prompt=self.custom_prompt)

                response = await self.handle_documents_and_search_mode(agent, message)

            else:
                response = await self.handle_default_mode(message, loaded_history)

            for token in response:
                yield history, messages_history

        except ValueError as e:
            gr.Warning(str(e))
        except Exception as e:
            gr.Warning(f"Unexpected error occurred: {str(e)}")

    def build_history_message(self, loaded_history):
        history_messages = []
        for history_data in loaded_history[-min(5, len(loaded_history)):]:
            history_messages.append(ChatMessage(role="user", content=history_data[0]))
            history_messages.append(ChatMessage(role="assistant", content=history_data[1]))
        return history_messages

    def build_tools(self, company, tender):
        tools = []
        if company:
            tools.append(QueryEngineTool(
                query_engine=company.as_query_engine(similarity_top_k=10),
                metadata=ToolMetadata(name='company_index', description=f'{self.company_description}')
            ))
        if tender:
            tools.append(QueryEngineTool(
                query_engine=tender.as_query_engine(similarity_top_k=10),
                metadata=ToolMetadata(name='tender_index', description=f'{self.tender_description}')
            ))
        return tools

    async def handle_only_document_mode(self, agent, message: str):
        qa_message = f"{message}. Devi rispondere in italiano."
        if self.colbert == 'No':
            response = agent.stream_chat(qa_message)
            stream_token = ""
            for token in response.response_gen:
                stream_token += token
                yield stream_token

            if stream_token and message:
                add_chat_history(f"#{len(self.source_infor_results)}:{message}::::{stream_token}", self.get_source_info(), self.current_session_id)
        else:
            ragatouille_pack = RAGatouilleRetrieverPack(
                self.documents,
                llm=OpenAI(model='gpt-4-1106-preview'),
                index_name="my_index",
                top_k=5
            )
            response = ragatouille_pack.run(qa_message)

            stream_token = ""
            for token in str(response):
                stream_token += token
                yield stream_token

            if stream_token and message:
                add_chat_history(f"#{len(self.source_infor_results)}:{message}::::{stream_token}", self.get_source_info(), self.current_session_id)

    async def handle_documents_and_search_mode(self, agent, message: str):
        qa_message = f":{message}. Devi rispondere in italiano."
        response = agent.stream_chat(qa_message)

        stream_token = ""
        if response.source_nodes:
            self.response_sources = response.source_nodes
        else:
            self.response_sources = "No sources found."

        for token in response.response_gen:
            stream_token += token
            yield stream_token

        if stream_token and message:
            add_chat_history(f"#{len(self.source_infor_results)}:{message}::::{stream_token}", self.get_source_info(), self.current_session_id)

    async def handle_default_mode(self, message, loaded_history):
        history_message = self.build_history_message(loaded_history)
        history_message.append({"role": "user", "content": message})
        qa_message = f"Devi rispondere in italiano."
        history_message.append({"role": "user", "content": qa_message})
        content = self.openai_agent(history_message)

        partial_message = ""
        for chunk in content:
            if chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

        if partial_message and message:
            add_chat_history(f"{message}::::{partial_message}", "no_data", self.current_session_id)

    def update_debug_info(self, upload_file):
        debug_info = self.status
        return debug_info

    def update_source_info(self):
        source_info = "Sources: \n"
        if self.response_sources == "No sources found.":
            return source_info
        else:
            for node_with_score in self.response_sources:
                node = node_with_score.node
                text = node.text
                extra_info = node.extra_info or {}
                file_name = extra_info.get("file_name")
                page_label = extra_info.get("page_label")
                source_info += f"File Name: {file_name}\n, Page Label: {page_label}\n\n"
            return source_info
        
    def update_url_info(self):
        directory_path = f"data/url/{self.current_session_id}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        self.google_source_urls = [['No data', 'No data', 'No data', 'No data', 'No data',
                        'No data', 'No data', 'No data', 'No data', 'No data', 'No data']]
        file_list = os.listdir(directory_path)
        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    self.google_source_urls = []
                    value = file_content.strip().split('\n')
                    value = value[:10]
                    for val in value:
                        self.google_upload_url += f"siteSearch={val}&"
                    if len(value) < 10:
                        value.extend(['No data'] * (10 - len(value)))
                    self.google_source_urls.append(['question']+value)

    def update_source(self):
        final_result = []
        self.source_infor_results = []
        
        try:
            with sqlite3.connect("chat_history.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT *
                    FROM chat_history
                    WHERE session_id = ?
                    ORDER BY id
                """, (self.current_session_id,))
                rows = cursor.fetchall()

                if rows:
                    source_informs = [
                        list(row[2].split("&&&&")) + [row[1].split("::::")[0]]
                        for row in rows if row[2] != "no_data"
                    ]

                    for source_inform in source_informs:
                        for temp in source_inform[:-1]:
                            temp_1 = list(temp.split("::::")) + [source_inform[-1]]
                            self.source_infor_results.append(temp_1)

                    final_result = [
                        [result[-1], result[0], result[1]]
                        for result in self.source_infor_results
                    ]

            if not final_result:
                final_result = [['No Data', 'No Data', 'No Data']]

        except sqlite3.Error as e:
            gr.Info(f"Database error: {str(e)}")
            final_result = [['No Data', 'No Data', 'No Data']]

        return final_result


    def get_source_info(self):
        source_info = ""
        if self.response_sources == "No sources found.":
            return source_info
        else:
            for node_with_score in self.response_sources:
                node = node_with_score.node
                text = node.text
                extra_info = node.extra_info or {}
                file_name = extra_info.get("file_name")
                page_label = extra_info.get("page_label")
                source_info += f"{file_name}::::{page_label}::::{text}&&&&"
            return source_info


    def delete_index(self, index_key):
        if openai.api_key:
            gr.Info("Deleting index..")
            directory_path = f"./storage/{index_key}/{self.current_session_id}"
            backup_path = f"./backup_path/{index_key}/{self.current_session_id}"
            documents_path = f"data/{index_key}/{self.current_session_id}"
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

            for item in self.doc_ids[index_key]:
                self.indices[index_key].delete_ref_doc(item, delete_from_docstore=True)

            self.status = ""
            gr.Info("Index is deleted")
            debug_info = self.status
            return debug_info
        else:
            return None


    def delete_row(self, index_key, file_name):
        current_datetime = datetime.now().timestamp()
        if openai.api_key:
            gr.Info("Deleting index..")
            backup_path = f"./backup_path/{index_key}/{self.current_session_id}/{current_datetime}"
            index_path = f"./storage/{self.current_session_id}/{index_key}"
            documents_path = f"./data/{index_key}/{self.current_session_id}/{file_name}"
            directory_path = f"data/{index_key}/{self.current_session_id}"
            if not os.path.exists(documents_path):
                os.makedirs(documents_path)
            if not os.path.exists(backup_path):
                os.makedirs(backup_path)
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            shutil.move(documents_path, backup_path)
            shutil.rmtree(index_path)

            string_to_match = f'data\\{index_key}\\{self.current_session_id}\\{file_name}'
            filtered_list = [item for item in self.doc_ids[index_key]
                            if item.startswith(string_to_match)]
            for item in filtered_list:
                self.indices[index_key].delete_ref_doc(item, delete_from_docstore=True)
            id = self.indices[index_key].index_id
            status = ""
            gr.Info("Index is deleted")
            debug_info = status
            return debug_info
        else:
            return None

    def get_sources(self, choice):
        if choice == "view source":
            return gr.Textbox.update(value=self.response_sources)

    def set_openai_api_key(self, api_key: str):
        if api_key:
            openai.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            llm_predictor = LLMPredictor(llm=ChatOpenAI(
                temperature=0, model_name=self.model, streaming=True))
            global service_context
            service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor, chunk_size=1024)
        else:
            gr.Warning("Please enter a valid OpenAI API key or set the env key.")

    def openai_agent(self, prompt):
        response = openai.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=1.0,
            stream=True
        )
        return response

    def update_company_info(self, upload_file):
        file_value = self.get_company_files_inform(directory_path=f"data/company/{self.current_session_id}")
        return gr.update(value=file_value)

    def update_tender_info(self, upload_file):
        file_value = self.get_tender_files_inform(directory_path=f"data/tender/{self.current_session_id}")
        return gr.update(value=file_value)

    def set_tender_pdf(self, evt: gr.SelectData):
        select_data = evt.index
        if select_data[1] == 0:
            pdf_viewer_content = f'<iframe src="file/data/tender/{self.current_session_id}/{evt.value}" width="100%" height="600px"></iframe>'
            file_path = self.search_files_by_name("./data", evt.value)
            webbrowser.open(file_path)
            return gr.update(value=pdf_viewer_content)
        else:
            self.delete_row("tender", self.file_tender_inform_datas[int(select_data[0])][0])

    def set_company_pdf(self, evt: gr.SelectData):
        select_data = evt.index
        if select_data[1] == 0:
            pdf_viewer_content = f'<iframe src="file/data/company/{self.current_session_id}/{evt.value}" width="100%" height="600px"></iframe>'
            file_path = self.search_files_by_name("./data", evt.value)
            webbrowser.open(file_path)
            return gr.update(value=pdf_viewer_content)
        else:
            self.delete_row(
                "company", self.file_company_inform_datas[int(select_data[0])][0])

    def set_source_pdf(self, evt: gr.SelectData):
        pdf_viewer_content = f'<iframe src="file/data/company/{evt.value}" width="100%" height="800px"></iframe>'
        return gr.update(value=pdf_viewer_content)

    def set_highlight_pdf(self, evt: gr.SelectData):
        select_data = evt.index
        file_name = self.source_infor_results[int(select_data[0])][0]
        source_text = self.source_infor_results[int(select_data[0])][2]
        file_path = self.search_files_by_name("./data", file_name)
        webbrowser.open(file_path)
        pdf_viewer_content = f'<h4>{source_text}</h4>'
        return gr.update(value=pdf_viewer_content)

    def search_files_by_name(self, root_dir, file_name):
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename == file_name:
                    return os.path.join(foldername, filename)

    def update_session(self, update_data):
        old_sessions = self.getSessionList()
        new_sessions = update_data.values

        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                for i in range(min(len(new_sessions), len(old_sessions))):
                    old_session_title = old_sessions[i][0]
                    new_session_title = new_sessions[i][0]
                    if old_session_title != new_session_title:
                        cursor.execute(
                            "UPDATE session_history SET session_title = ? WHERE id = ?;",
                            (new_session_title, self.session_list[i])
                        )
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database Error: {e}")

    def add_session(self, session_title):
        if not session_title:
            gr.Info("You have to input a session title.")
            return gr.update(value=self.getSessionList())

        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO session_history(session_title) VALUES (?)",
                    (session_title,)
                )
                self.current_session_id = cursor.lastrowid
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database Error: {e}")
            gr.Info("Failed to add session due to a database error.")
        
        return gr.update(value=self.getSessionList())

    def set_session(self, evt: gr.SelectData):
        try:
            select_data = evt.index
            self.current_session_id = self.session_list[int(select_data[0])]
        except IndexError as e:
            print(f"Selection Error: {e}")
            gr.Info("Invalid session selection.")

ragBot = RagBot()

with open(
    "./assets/custom.css",
    "r",
    encoding="utf-8",
) as f:
    customCSS = f.read()

title = f"""<h2 align="center">BLM Mostro <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACiklEQVR4AWIYSoAP0Hw5wMoRhVE4i8zOvtq2bSNOjbC2jaC27Tasbdt2g9q27duT5Ezy93a9M807ybd392LO9fwLmoIF4AJ4BX6Bb+AROAAmgOput9trp3F2MA98AkpgGb8BSuMO6O5yucyYXTmK/uALUBytIo+0UfZkvs4NUD16d5crJT73AMWRD8ZoUiPdxLwNQKqsqFuRpidEx/tG7E2jC2z8DOQVZQlIjoH+2myZNK9r5Xk8HjeSWUCRcRGYuw0kh0SjLzBNHqCD+YGuikDXJKAE3UFITQBKoymIWj6fz83NqAQ/uFwBVZLr9QIUBW3BNrAMxKLS3IQTaDqFnbiA5Ql4TLewQmttfY1onblUXp/PdIvfJpJb9Gis18/PghvsnVNqTZ9TesFQFrzjZrJdmAEDyQqgSF5ZfkLufFDfTnOepH36iZA33hd5y4E1aJTSxj60BefAN+GzyCrMyoxzMMV358SN2J5+R+TxU1yf/6Hi9LuSaDqQnRlnMEUZnXTmndL2ryWAqb4J74FVNm/C1uCE5rNEVjilHcOGwDZxMAeAEvSUdUaJi6iqgydgHVCkoCwvzMxrfI87fRVfME3zH59dLGweIDSLWvo7RXsZtQ7UpryIggqyIxvAkjhex1e4vCVFrHHJFWJQs+wKSEzTHygg+QUqh9soZ2TorYdk3NF5A8+gJgYhgv4grDKCK2I5cmodPKQ/iBfMB1DTyvN6vW4k04T5LL8/IeINnp7Rr+KDB3Lk64KE5aVF3fKgstWejDAMI2JzGSGPAT8i+GPSnfl6vQeclbhUECwTHbH4xGv7mTQlzzhrSeM115elM5fhhhZcfGDAMQ/UZfjlrNwQlBQXjhnPc/4Aqf7wDR6odCYAAAAASUVORK5CYII=" alt="client" style="display: inline;"></h2>"""

clear = gr.Button("🧹 Start fresh")

with gr.Blocks(css=customCSS, theme=WORDLIFT_THEME) as demo:
    gr.Info("Please enter a valid OpenAI API key or set the env key.")
    gr.HTML(value=title)
    session_state = gr.State([])
    def user(user_message, history):
        return "", history + [[user_message, None]]

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            session_title = gr.Textbox(label="Session Title")
            new_session_btn = gr.Button(value="New Session")
            session_list_dataframe = gr.Dataframe(
                value=ragBot.getSessionList,
                headers=["Session List"],
                datatype=["str"],
                col_count=(1, "fixed"),
                height=600,
                interactive=True,
                elem_id="session_dataframe"
            )
        with gr.Column(scale=15):
            with gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(value=get_chat_history(ragBot.current_session_id), elem_id="chuanhu_chatbot", height="850px")
                    msg = gr.Textbox(
                        label=" 🪄",
                        placeholder="Type a message to the bot and press enter",
                        container=False,
                    )
                with gr.Column(scale=4):
                    source_dataframe = gr.Dataframe(
                        value=ragBot.update_source,
                        headers=["question", "File name", "page numeber"],
                        datatype=["str", "str", "str"],
                        col_count=(3, "fixed"),
                        row_count=5,
                        wrap=False,
                        height=300,
                        label="Agent Mode Q&A history",
                        interactive=False,
                        elem_id="source_dataframe"
                    )
                    pdf_viewer_html = gr.HTML(value=pdf_view_url, label="preview", elem_id="pdf_reference")
                    clear = gr.Button("🧹 Start fresh")
            with gr.Accordion("⚙️ Settings", open=False):
                with gr.Tab("history"):
                    google_search_dataframe = gr.Dataframe(
                        value=ragBot.google_source_urls,
                        headers=["Question", "Source URL-1", "Source URL-2", "Source URL-3", "Source URL-4",
                                 "Source URL-5", "Source URL-6", "Source URL-7", "Source URL-8", "Source URL-9", "Source URL-10"],
                        datatype=["str", "str", "str", "str", "str","str", "str", "str", "str", "str", "str"],
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
                openai_api_key_textbox.change(ragBot.set_openai_api_key, inputs=openai_api_key_textbox)
                chatting_mode_radio = gr.Radio(
                    value="Documents and Search", choices=["Only Document", "No Documents", "Documents and Search"], label="Chatting Quality > Agent Type"
                )
                custom_prompt = gr.Textbox(
                    placeholder="Here goes the custom prompt",
                    value=TEMPLATE,
                    lines=5,
                    label="Custom Prompt",
                )
                custom_prompt.change(fn=ragBot.set_prompt, inputs=custom_prompt)
                tender_description_textbox = gr.Textbox(
                    placeholder="Here goes the tender description",
                    value=ragBot.tender_description,
                    lines=5,
                    label="Tender Description",
                )
                tender_description_textbox.change(lambda x: ragBot.set_description("tender_description", x), inputs=tender_description_textbox)

                company_description_textbox = gr.Textbox(
                    placeholder="Here goes the company description",
                    value=ragBot.company_description,
                    lines=5,
                    label="Company Description",
                )

                company_description_textbox.change(lambda x: ragBot.set_description("company_description", x), inputs=company_description_textbox)
                radio = gr.Radio(
                    value="gpt-4-turbo-preview", choices=["gpt-3.5-turbo", "gpt-4-turbo-preview"], label="Models"
                )
                radio.change(ragBot.set_model, inputs=radio)
                radioColBERT = gr.Radio(
                    value="No", choices=["Yes", "No"], label="ColBERT"
                )
                radioColBERT.change(ragBot.set_colbert, inputs=radioColBERT) 
                with gr.Row():
                    tender_data = ragBot.get_tender_files_inform(directory_path=f"data/tender/{ragBot.current_session_id}")
                    company_data = ragBot.get_company_files_inform(directory_path=f"data/company/{ragBot.current_session_id}")

                    tender_dataframe = gr.Dataframe(
                        value=tender_data,
                        headers=["Tender File Name", "Action"],
                        datatype=["str", "str"],
                        col_count=(2, "fixed"),
                        label="Tender File list",
                        interactive=False
                    )

                    company_dataframe = gr.Dataframe(
                        value=company_data,
                        headers=["Company File Name", "Action"],
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
            with gr.Accordion("🔍 Debug", open=False):
                debug_output = gr.Textbox(
                    placeholder="Debug output will be printed here",
                    lines=5,
                    label="Debug Output",
                )

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        ragBot.bot, [chatbot, session_state], [chatbot, session_state]).then(
        lambda: gr.update(value=get_chat_history(ragBot.current_session_id)), None, outputs=[chatbot]).then(
        lambda: gr.update(value=ragBot.update_source()), None, outputs=source_dataframe).then(
            ragBot.update_source_info, None, outputs=sources
    ).then(
        lambda: gr.update(value=ragBot.google_source_urls), None, outputs=google_search_dataframe)

    file_response1 = upload_button1.upload(
        lambda files: ragBot.upload_file(files, "tender"), upload_button1
    ).then(
        ragBot.update_tender_info, inputs=[tender_dataframe], outputs=tender_dataframe
    )
    file_response2 = upload_button2.upload(
        lambda files: ragBot.upload_file(files, "company"), upload_button2
    ).then(
        ragBot.update_company_info, inputs=[
            company_dataframe], outputs=company_dataframe
    )
    file_response3 = upload_button3.upload(
        lambda files: ragBot.upload_file(files, "url"), upload_button3
    ).then(
        lambda: gr.update(value=ragBot.google_source_urls), None, outputs=google_search_dataframe)
    file_response1.then(
        ragBot.update_debug_info, inputs=[upload_button1], outputs=debug_output
    )
    file_response2.then(
        ragBot.update_debug_info, inputs=[upload_button2], outputs=debug_output
    )

    delete_button1.click(lambda: ragBot.delete_index("tender")).then(
        ragBot.update_tender_info, inputs=[tender_dataframe], outputs=tender_dataframe
    )
    delete_button2.click(lambda: ragBot.delete_index("company")).then(
        ragBot.update_company_info, inputs=[
            company_dataframe], outputs=company_dataframe
    )

    chatting_mode_radio.change(ragBot.set_chatting_mode, inputs=[chatting_mode_radio])

    clear.click(lambda: [], None, chatbot, queue=False).then(
        ragBot.clear_chat_history, None, outputs=[chatbot]).then(
        lambda: gr.update(value=ragBot.update_source()), None, outputs=source_dataframe).then(
        lambda: gr.update(value=ragBot.getSessionList()), None, outputs=session_list_dataframe).then(
        lambda: gr.update(value=ragBot.google_source_urls), None, outputs=google_search_dataframe)

    tender_dataframe.select(ragBot.set_tender_pdf, None, pdf_viewer_html).then(
        ragBot.update_tender_info, inputs=[tender_dataframe], outputs=tender_dataframe
    )
    company_dataframe.select(ragBot.set_company_pdf, None, pdf_viewer_html).then(
        ragBot.update_company_info, inputs=[
            company_dataframe], outputs=company_dataframe
    )
    source_dataframe.select(ragBot.set_highlight_pdf, None, pdf_viewer_html)

    session_list_dataframe.select(ragBot.set_session, None, None).then(
        lambda: gr.update(value=get_chat_history(ragBot.current_session_id)), None, outputs=[chatbot]).then(
        lambda: gr.update(value=ragBot.update_source()), None, outputs=source_dataframe).then(
            ragBot.update_source_info, None, outputs=sources).then(
            ragBot.update_url_info, None, outputs=google_search_dataframe).then(
        lambda: gr.update(value=ragBot.google_source_urls), None, outputs=google_search_dataframe).then(
            ragBot.update_tender_info, inputs=[tender_dataframe], outputs=tender_dataframe).then(
            ragBot.update_company_info, inputs=[company_dataframe], outputs=company_dataframe)

    session_list_dataframe.input(ragBot.update_session, inputs=[
                                 session_list_dataframe])

    new_session_btn.click(ragBot.add_session, inputs=[session_title], outputs=[
                          session_list_dataframe])

demo.queue().launch(inline=True).then(lambda: gr.update(
    value=get_chat_history(ragBot.current_session_id)), None, outputs=[chatbot])