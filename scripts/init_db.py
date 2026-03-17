import sqlite3
from pathlib import Path

DB_PATH = Path("db/app.db")
SCHEMA_PATH = Path("db/schema.sql")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")