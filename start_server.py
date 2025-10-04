import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
import uvicorn

def clear_database_on_startup():
    """Clear database on server startup to ensure clean state"""
    try:
        from app.database import Neo4jConnection
        print("\n" + "="*60)
        print("CLEARING DATABASE ON STARTUP")
        print("="*60)
        
        neo4j = Neo4jConnection()
        neo4j.connect()
        
        if neo4j.driver:
            with neo4j.driver.session() as session:
                # Count nodes before deletion
                count_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = count_result.single()["count"]
                
                if node_count > 0:
                    print(f"Found {node_count} nodes in database")
                    
                    # Delete with explicit transaction
                    with session.begin_transaction() as tx:
                        tx.run("MATCH (n) DETACH DELETE n")
                        tx.commit()
                    
                    # Verify deletion
                    verify_result = session.run("MATCH (n) RETURN count(n) as remaining")
                    remaining = verify_result.single()["remaining"]
                    
                    if remaining == 0:
                        print(f"✅ Successfully cleared all {node_count} nodes from Neo4j")
                    else:
                        print(f"⚠️ Warning: {remaining} nodes still remain")
                else:
                    print("Database is already empty")
        
        neo4j.close()
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"⚠️ Could not clear database on startup: {e}")
        print("Server will continue, but old data may still be present\n")

if __name__ == "__main__":
    # Clear database before starting server
    print("🗑️ Cleaning up old data...")
    clear_database_on_startup()
    
    print("🚀 Starting GraphRAG Chatbot server...")
    print("📱 Open http://localhost:8000 in your browser")
    print("🎨 Open chatbot_ui.html to start chatting")
    print("ℹ️  Press Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")