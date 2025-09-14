import sqlite3

conn = sqlite3.connect("sqlite.db")
cur = conn.cursor()

# ❌ Drop table if it already exists
cur.execute("DROP TABLE IF EXISTS students")

# ✅ Create new table
cur.execute("""
CREATE TABLE students (
    id INTEGER,
    name TEXT NOT NULL,
    age INTEGER
)
""")

conn.commit()
conn.close()

