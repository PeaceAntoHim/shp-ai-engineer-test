import psycopg2
import json
from dotenv import load_dotenv
load_dotenv()
import os

class Database:
    """Simple database connection and operations"""
    
    def __init__(self):
        self.connection = None
        self.connect()
        self.create_table()
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS"),
                port=os.getenv("DB_PORT")
            )
            print("Database connected successfully")
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise
    
    def create_table(self):
        """Create vectors table if not exists"""
        cursor = self.connection.cursor()
        
        # Create schema
        cursor.execute("CREATE SCHEMA IF NOT EXISTS vector_db")
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_db.vectors (
                id SERIAL PRIMARY KEY,
                embedding REAL[] NOT NULL,
                content TEXT,
                metadata JSONB DEFAULT '{}'
            )
        """)
        
        self.connection.commit()
        cursor.close()
        print("Database table ready")
    
    def insert_vector(self, embedding, content=None, metadata=None):
        """Insert a vector into database"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO vector_db.vectors (embedding, content, metadata)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (embedding, content, json.dumps(metadata or {})))
        
        vector_id = cursor.fetchone()[0]
        self.connection.commit()
        cursor.close()
        return vector_id
    
    def get_all_vectors(self):
        """Get all vectors from database"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT id, embedding, content, metadata 
            FROM vector_db.vectors
        """)
        
        results = cursor.fetchall()
        cursor.close()
        
        vectors = []
        for row in results:
            vectors.append({
                'id': row[0],
                'embedding': list(row[1]),  # Convert array to list
                'content': row[2],
                'metadata': row[3]
            })
        
        return vectors
    
    def get_vector(self, vector_id):
        """Get single vector by ID"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT id, embedding, content, metadata 
            FROM vector_db.vectors 
            WHERE id = %s
        """, (vector_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            return {
                'id': row[0],
                'embedding': list(row[1]),
                'content': row[2],
                'metadata': row[3]
            }
        return None
    
    def delete_vector(self, vector_id):
        """Delete vector by ID"""
        cursor = self.connection.cursor()
        
        cursor.execute("DELETE FROM vector_db.vectors WHERE id = %s", (vector_id,))
        deleted = cursor.rowcount > 0
        
        self.connection.commit()
        cursor.close()
        return deleted
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")
