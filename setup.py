#!/usr/bin/env python3
"""
Complete setup script for GraphRAG Chatbot
This script will set up and test the entire system
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path

def print_step(message):
    print(f"\n🔧 {message}")
    print("=" * 50)

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_warning(message):
    print(f"⚠️  {message}")

def check_python_packages():
    """Check if all required packages are installed"""
    print_step("Checking Python packages...")
    
    required_packages = [
        'neo4j', 'psycopg2-binary', 'fastapi', 'uvicorn', 
        'python-dotenv', 'ollama', 'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'psycopg2-binary':
                __import__('psycopg2')
            else:
                __import__(package.replace('-', '_'))
            print_success(f"{package} is installed")
        except ImportError:
            missing_packages.append(package)
            print_error(f"{package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print_success("All packages installed successfully")
        except subprocess.CalledProcessError:
            print_error("Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def create_env_file():
    """Create .env file with default configuration"""
    print_step("Creating .env file...")
    
    env_content = """# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=graphrag
POSTGRES_USER=admin
POSTGRES_PASSWORD=password123

# Ollama Configuration
MODEL_NAME=gemma:1b
OLLAMA_BASE_URL=http://localhost:11434
"""
    
    if os.path.exists('.env'):
        print_warning(".env file already exists. Skipping...")
        return True
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print_success(".env file created")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False

def check_docker():
    """Check if Docker is running"""
    print_step("Checking Docker...")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_success(f"Docker is available: {result.stdout.strip()}")
            return True
        else:
            print_error("Docker is not available")
            return False
    except FileNotFoundError:
        print_error("Docker is not installed")
        return False

def start_databases():
    """Start Neo4j and PostgreSQL using Docker Compose"""
    print_step("Starting databases...")
    
    if not check_docker():
        print_warning("Skipping database setup - Docker not available")
        return False
    
    try:
        # Check if docker-compose.yml exists
        if not os.path.exists('docker-compose.yml'):
            print_error("docker-compose.yml not found")
            return False
        
        subprocess.check_call(['docker-compose', 'up', '-d'])
        print_success("Databases started successfully")
        
        # Wait for databases to be ready
        print("⏳ Waiting for databases to be ready...")
        time.sleep(10)
        
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to start databases: {e}")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    print_step("Checking Ollama...")
    
    try:
        import ollama
        
        # Try to list models to check if Ollama is running
        models = ollama.list()
        print_success("Ollama is running")
        
        # Check if required model is available
        model_name = os.getenv('MODEL_NAME', 'gemma:1b')
        available_models = [model['name'] for model in models.get('models', [])]
        
        if any(model_name in model for model in available_models):
            print_success(f"Model {model_name} is available")
            return True
        else:
            print_warning(f"Model {model_name} not found. Available models: {available_models}")
            print(f"📥 Pulling {model_name}...")
            try:
                ollama.pull(model_name)
                print_success(f"Model {model_name} downloaded successfully")
                return True
            except Exception as e:
                print_error(f"Failed to download model: {e}")
                return False
                
    except Exception as e:
        print_error(f"Ollama is not available: {e}")
        print("Please install Ollama from https://ollama.ai/")
        return False

def test_system():
    """Test the complete system"""
    print_step("Testing the system...")
    
    # Wait a bit for the server to start
    time.sleep(3)
    
    try:
        # Test server status
        response = requests.get('http://localhost:8000/status', timeout=10)
        if response.status_code == 200:
            status = response.json()
            print_success("Server is running")
            print(f"📊 System Status:")
            print(f"   Neo4j: {status.get('neo4j', 'unknown')}")
            print(f"   PostgreSQL: {status.get('postgresql', 'unknown')}")
            print(f"   Ollama: {status.get('ollama', 'unknown')}")
            print(f"   Model: {status.get('model', 'unknown')}")
        else:
            print_error(f"Server returned status {response.status_code}")
            return False
            
        # Add sample data
        print("\n📚 Adding sample documents...")
        response = requests.post('http://localhost:8000/add-sample-data', timeout=30)
        if response.status_code == 200:
            result = response.json()
            print_success(result['message'])
        else:
            print_warning("Failed to add sample data, but server is running")
        
        # Test chat functionality
        print("\n💬 Testing chat functionality...")
        test_question = "What is Artificial Intelligence?"
        response = requests.post(
            'http://localhost:8000/chat',
            json={'question': test_question},
            timeout=30
        )
        
        if response.status_code == 200:
            chat_result = response.json()
            print_success("Chat functionality is working")
            print(f"Question: {chat_result['question']}")
            print(f"Answer: {chat_result['answer'][:100]}...")
            if chat_result.get('context'):
                print(f"Context sources: {len(chat_result['context'])}")
        else:
            print_error(f"Chat test failed: {response.status_code}")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to connect to server: {e}")
        return False

def create_project_structure():
    """Ensure proper project structure exists"""
    print_step("Setting up project structure...")
    
    # Create necessary directories
    directories = ['app', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print_success(f"Directory {directory} ready")
    
    # Create __init__.py files
    init_files = ['app/__init__.py']
    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print_success(f"Created {init_file}")

def main():
    """Main setup function"""
    print("🚀 GraphRAG Chatbot Setup")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = True
    
    # Step 1: Check project structure
    create_project_structure()
    
    # Step 2: Check Python packages
    if not check_python_packages():
        success = False
    
    # Step 3: Create .env file
    if not create_env_file():
        success = False
    
    # Step 4: Check Ollama
    if not check_ollama():
        success = False
        print_warning("The system will work with limited functionality without Ollama")
    
    # Step 5: Start databases
    if not start_databases():
        success = False
        print_warning("The system will work with limited functionality without databases")
    
    if success or input("\n❓ Continue with setup despite some issues? (y/n): ").lower() == 'y':
        print_step("Starting the GraphRAG server...")
        
        # Create a simple test script
        test_script = '''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting GraphRAG Chatbot server...")
    print("📱 Open http://localhost:8000 in your browser")
    print("🎨 Open the HTML UI file to start chatting")
    print("⏹️  Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
'''
        
        with open('start_server.py', 'w') as f:
            f.write(test_script)
        
        print_success("Setup completed!")
        print("\n📋 Next Steps:")
        print("1. Run: python start_server.py")
        print("2. Open the chatbot_ui.html file in your browser")
        print("3. Start chatting with your GraphRAG assistant!")
        print("\n🔗 Useful URLs:")
        print("   • API: http://localhost:8000")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Neo4j Browser: http://localhost:7474 (neo4j/password123)")
        
        if input("\n🚀 Start the server now? (y/n): ").lower() == 'y':
            os.system('python start_server.py')
    else:
        print("❌ Setup cancelled due to errors")

if __name__ == "__main__":
    main()