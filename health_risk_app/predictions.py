import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    risk TEXT,
    timestamp TEXT,
    report_data TEXT
               )
               """)
conn.commit()
conn.close()

