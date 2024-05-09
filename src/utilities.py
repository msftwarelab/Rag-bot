import psutil
import os

def pdf_view_url():
  pdf_viewer_html = f'<iframe src="file/assets/pdf_viewer.html" width="100%" height="470px"></iframe>'
  return pdf_viewer_html

def get_available_storage():
  disk_usage = psutil.disk_usage('/')
  available_storage_gb = disk_usage.free / (2**30)
  return f"Available Storage: {available_storage_gb:.2f} GB"

def check_or_create_directory(directory_path):
  if not os.path.exists(directory_path):
      os.makedirs(directory_path)