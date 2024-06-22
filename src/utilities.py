import psutil
import os
import groupdocs_conversion_cloud
import shutil
from .config import GROUPDOCS_CLIENT_ID, GROUPDOCS_CLIENT_SECRET, HTML_OUTPUT_PATH

def convert_docx_to_html(file_path):
  print(f'Started converting {file_path} to html..')
  if os.path.exists(HTML_OUTPUT_PATH):
      shutil.rmtree(HTML_OUTPUT_PATH)
  os.makedirs(HTML_OUTPUT_PATH)

  convert_api = groupdocs_conversion_cloud.ConvertApi.from_keys(GROUPDOCS_CLIENT_ID, GROUPDOCS_CLIENT_SECRET)
  request = groupdocs_conversion_cloud.ConvertDocumentDirectRequest("html", file_path)
  response = convert_api.convert_document_direct(request)
  print(f"Got the {response} as a response")
  pre, ext = os.path.splitext(response)
  new_file_path = pre + ".html"

  # Check if the target .html file already exists
  if os.path.exists(new_file_path):
      print(f"File {new_file_path} already exists. Deleting existing file.")
      os.remove(new_file_path)

  os.rename(response, new_file_path)
  print(f"New response: {new_file_path}")

  if not os.path.isfile(new_file_path):
      raise FileNotFoundError(f"Converted file not found at: {new_file_path}")
  
  filename = os.path.basename(new_file_path)
  final_path = os.path.join(HTML_OUTPUT_PATH, filename)

  shutil.move(new_file_path, HTML_OUTPUT_PATH)

  return final_path

def get_available_storage():
  disk_usage = psutil.disk_usage('/')
  available_storage_gb = disk_usage.free / (2**30)
  return f"Available Storage: {available_storage_gb:.2f} GB"

def check_or_create_directory(directory_path):
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)