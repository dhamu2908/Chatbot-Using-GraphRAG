from neo4j import GraphDatabase
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None
        
    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("Connected to Neo4j successfully!")
            self.initialize_schema()
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
    
    def initialize_schema(self):
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
    
    def close(self):
        if self.driver:
            self.driver.close()

class PostgresConnection:
    def __init__(self):
        self.conn = None
        
    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                database=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD")
            )
            print("Connected to PostgreSQL successfully!")
            self.initialize_schema()
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
    
    def initialize_schema(self):
        queries = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        with self.conn.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
            self.conn.commit()
    
    def close(self):
        if self.conn:
            self.conn.close()