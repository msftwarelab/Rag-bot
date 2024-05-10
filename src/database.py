# database.py
import sqlite3
from .config import DATABASE_PATH

def create_tables():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor_1 = conn.cursor()
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

def add_chat_history(value, source_inform, session_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history(chat_history,source_inform,session_id) VALUES (?,?,?)", (value, source_inform, session_id))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chat_history
        FROM chat_history
        WHERE session_id = ?
        ORDER BY id
    """, (session_id,))
    rows = cursor.fetchall()
    chat_history = []
    if rows:
        for row in rows:
            chat_history.append(list(row[0].split("::::")))
    conn.close()
    return chat_history

def delete_chat_history(session_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM session_history WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
