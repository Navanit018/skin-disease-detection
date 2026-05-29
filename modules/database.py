import sqlite3
import json
import os
from pathlib import Path

DB_FILE = Path(__file__).parent.parent / 'skin_disease_history.db'
JSON_FILE = Path(__file__).parent.parent / 'prediction_history.json'

def init_db():
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            type TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(datetime_str, type_str, disease_str, confidence_val, severity_str, details_dict=None):
    init_db()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    
    details_str = json.dumps(details_dict) if details_dict else None
    
    cursor.execute('''
        INSERT INTO prediction_history (datetime, type, disease, confidence, severity, details)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime_str, type_str, disease_str, confidence_val, severity_str, details_str))
    
    conn.commit()
    conn.close()

def load_history():
    init_db()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM prediction_history ORDER BY id ASC')
    rows = cursor.fetchall()
    
    history = []
    for row in rows:
        item = {
            'id': row['id'],
            'datetime': row['datetime'],
            'type': row['type'],
            'disease': row['disease'],
            'confidence': row['confidence'],
            'severity': row['severity']
        }
        if row['details']:
            try:
                item['details'] = json.loads(row['details'])
            except Exception:
                item['details'] = row['details']
        history.append(item)
        
    conn.close()
    return history

def migrate_json_to_sqlite():
    if not JSON_FILE.exists():
        return
    
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            return
            
        init_db()
        conn = sqlite3.connect(str(DB_FILE))
        cursor = conn.cursor()
        
        # Check if we already migrated or populated SQLite to avoid duplicates
        cursor.execute('SELECT COUNT(*) FROM prediction_history')
        count = cursor.fetchone()[0]
        if count > 0:
            conn.close()
            return  # Already migrated or populated
            
        for item in data:
            datetime_str = item.get('datetime', '')
            type_str = item.get('type', 'disease')
            disease_str = item.get('disease', '')
            confidence_val = item.get('confidence', 0.0)
            severity_str = item.get('severity', '')
            
            # Put any other fields into details
            details_keys = [k for k in item.keys() if k not in ['datetime', 'type', 'disease', 'confidence', 'severity']]
            details_dict = {k: item[k] for k in details_keys} if details_keys else None
            details_str = json.dumps(details_dict) if details_dict else None
            
            cursor.execute('''
                INSERT INTO prediction_history (datetime, type, disease, confidence, severity, details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime_str, type_str, disease_str, confidence_val, severity_str, details_str))
            
        conn.commit()
        conn.close()
        print(f"Successfully migrated {len(data)} items from JSON to SQLite database.")
        
        # Optionally rename the JSON file to mark it migrated
        JSON_FILE.rename(JSON_FILE.with_suffix('.json.bak'))
        
    except Exception as e:
        print(f"Error migrating JSON to SQLite: {e}")

# Run automatic migration on import
init_db()
migrate_json_to_sqlite()
