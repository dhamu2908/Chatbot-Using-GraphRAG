import ollama
import os
from typing import List, Dict, Tuple, Optional
from database import Neo4jConnection
import hashlib
import re
import numpy as np
from collections import Counter

class GraphRAGSystem:
    def __init__(self):
        self.neo4j = Neo4jConnection()
        try:
            self.neo4j.connect()
        except Exception as e:
            print(f"Neo4j connection warning: {e}")
        self.model_name = os.getenv("MODEL_NAME", "gemma:1b")
        self.embedding_model = "nomic-embed-text"
        self._ensure_embedding_model()
    
    def _ensure_embedding_model(self):
        """Ensure embedding model is available"""
        try:
            ollama.embeddings(model=self.embedding_model, prompt="test")
        except:
            print(f"Pulling embedding model...")
            os.system(f"ollama pull {self.embedding_model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate semantic embedding"""
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text.strip()[:1000]
            )
            return response['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        if not vec1 or not vec2:
            return 0.0
        v1, v2 = np.array(vec1), np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def understand_query(self, question: str) -> Dict:
        """
        Analyze query intent and expand it
        Returns: {
            'original': original question,
            'intent': question type,
            'expanded': rephrased versions,
            'key_entities': important terms
        }
        """
        prompt = f"""Analyze this question and help improve search:

Question: "{question}"

Provide:
1. Question type (who/what/when/where/why/how/explanation)
2. Key entities or concepts (names, places, terms)
3. 2 alternative ways to phrase this question
4. Search keywords (5-8 words)

Format your response as:
Type: [type]
Entities: [comma-separated]
Rephrase 1: [alternative question]
Rephrase 2: [alternative question]
Keywords: [comma-separated]"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.3, 'num_predict': 150}
            )
            
            if response and 'response' in response:
                text = response['response']
                
                # Parse response
                intent = 'unknown'
                entities = []
                rephrases = [question]
                keywords = []
                
                for line in text.split('\n'):
                    line = line.strip()
                    if line.startswith('Type:'):
                        intent = line.split(':', 1)[1].strip().lower()
                    elif line.startswith('Entities:'):
                        entities = [e.strip() for e in line.split(':', 1)[1].split(',')]
                    elif line.startswith('Rephrase'):
                        rephrase = line.split(':', 1)[1].strip()
                        if rephrase:
                            rephrases.append(rephrase)
                    elif line.startswith('Keywords:'):
                        keywords = [k.strip() for k in line.split(':', 1)[1].split(',')]
                
                return {
                    'original': question,
                    'intent': intent,
                    'entities': entities[:5],
                    'expanded': rephrases[:3],
                    'keywords': keywords[:8]
                }
        except Exception as e:
            print(f"Query understanding failed: {e}")
        
        # Fallback to simple extraction
        return {
            'original': question,
            'intent': 'unknown',
            'entities': self.extract_entities_simple(question),
            'expanded': [question],
            'keywords': self.extract_keywords(question)
        }
    
    def extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction"""
        entities = []
        # Proper nouns
        proper = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend([p.lower() for p in proper])
        # Acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        entities.extend([a.lower() for a in acronyms])
        return list(set(entities))[:5]
    
    def chunk_text(self, text: str) -> List[str]:
        """Intelligent chunking"""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current = []
        word_count = 0
        
        for sent in sentences:
            words = len(sent.split())
            if word_count + words > 150 and current:
                chunks.append('. '.join(current) + '.')
                current = [current[-1]] if current else []
                word_count = len(current[0].split()) if current else 0
            current.append(sent)
            word_count += words
        
        if current:
            chunks.append('. '.join(current) + '.')
        
        return chunks if chunks else [text]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords"""
        keywords = set()
        proper = re.findall(r'\b[A-Z][a-z]+\b', text)
        keywords.update(p.lower() for p in proper if len(p) > 2)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        keywords.update(a.lower() for a in acronyms)
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
            'what', 'who', 'when', 'where', 'why', 'how'
        }
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = Counter(w for w in words if w not in stop_words)
        keywords.update([w for w, c in word_freq.most_common(8)])
        
        return list(keywords)[:15]
    
    def store_document(self, title: str, content: str):
        """Store document with embeddings"""
        if not hasattr(self.neo4j, 'driver') or not self.neo4j.driver:
            print("Neo4j not available")
            return
        
        doc_id = hashlib.md5(f"{title}{content}".encode()).hexdigest()[:16]
        chunks = self.chunk_text(content)
        
        print(f"\nStoring: {title} ({len(chunks)} chunks)")
        
        try:
            with self.neo4j.driver.session() as session:
                session.run(
                    "MERGE (d:Document {id: $id, title: $title})",
                    id=doc_id, title=title
                )
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    embedding = self.get_embedding(chunk)
                    if not embedding:
                        continue
                    
                    keywords = self.extract_keywords(chunk)
                    
                    session.run(
                        """
                        MERGE (c:Chunk {
                            id: $id, content: $content, document_id: $doc_id,
                            embedding: $embedding
                        })
                        """,
                        id=chunk_id, content=chunk, doc_id=doc_id, embedding=embedding
                    )
                    
                    session.run(
                        "MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id}) "
                        "MERGE (d)-[:CONTAINS]->(c)",
                        doc_id=doc_id, chunk_id=chunk_id
                    )
                    
                    for kw in keywords:
                        if kw and len(kw) > 1:
                            session.run("MERGE (k:Keyword {name: $keyword})", keyword=kw)
                            session.run(
                                "MATCH (c:Chunk {id: $chunk_id}), (k:Keyword {name: $keyword}) "
                                "MERGE (c)-[:HAS_KEYWORD]->(k)",
                                chunk_id=chunk_id, keyword=kw
                            )
                
                print(f"  Stored successfully")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def retrieve_with_reranking(self, question: str, query_analysis: Dict, 
                                initial_k: int = 15, final_k: int = 5) -> Tuple[List[str], List[Dict]]:
        """
        Two-stage retrieval with re-ranking
        Stage 1: Get initial candidates (fast)
        Stage 2: Re-rank using query understanding (accurate)
        """
        if not hasattr(self.neo4j, 'driver') or not self.neo4j.driver:
            return (["Database not available."], [])
        
        print(f"\nQuery Analysis:")
        print(f"  Intent: {query_analysis['intent']}")
        print(f"  Entities: {query_analysis['entities']}")
        print(f"  Keywords: {query_analysis['keywords']}")
        
        # Generate embeddings for all query variations
        query_embeddings = []
        for q in query_analysis['expanded']:
            emb = self.get_embedding(q)
            if emb:
                query_embeddings.append(emb)
        
        if not query_embeddings:
            return (["Failed to generate embeddings."], [])
        
        try:
            with self.neo4j.driver.session() as session:
                # Stage 1: Get initial candidates with keyword filter
                all_keywords = query_analysis['keywords'] + query_analysis['entities']
                
                if all_keywords:
                    result = session.run(
                        """
                        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[:HAS_KEYWORD]->(k:Keyword)
                        WHERE k.name IN $keywords
                        WITH DISTINCT c, d
                        RETURN c.id as chunk_id, c.content as content, c.embedding as embedding,
                               d.title as doc_title, d.id as doc_id
                        """,
                        keywords=all_keywords
                    )
                    candidates = list(result)
                    
                    if len(candidates) < 10:
                        result = session.run(
                            """
                            MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
                            RETURN c.id as chunk_id, c.content as content, c.embedding as embedding,
                                   d.title as doc_title, d.id as doc_id
                            """
                        )
                        candidates = list(result)
                else:
                    result = session.run(
                        """
                        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
                        RETURN c.id as chunk_id, c.content as content, c.embedding as embedding,
                               d.title as doc_title, d.id as doc_id
                        """
                    )
                    candidates = list(result)
                
                print(f"\nStage 1: Found {len(candidates)} candidate chunks")
                
                # Stage 2: Calculate similarity with all query variations
                scored_chunks = []
                for record in candidates:
                    chunk_embedding = record["embedding"]
                    if not chunk_embedding:
                        continue
                    
                    # Calculate max similarity across all query variations
                    similarities = [
                        self.cosine_similarity(qe, chunk_embedding) 
                        for qe in query_embeddings
                    ]
                    max_similarity = max(similarities)
                    avg_similarity = np.mean(similarities)
                    
                    # Boost score if entities match
                    content_lower = record["content"].lower()
                    entity_boost = sum(1 for e in query_analysis['entities'] 
                                     if e.lower() in content_lower) * 0.1
                    
                    final_score = max_similarity + entity_boost
                    
                    scored_chunks.append({
                        "chunk_id": record["chunk_id"],
                        "content": record["content"],
                        "doc_title": record["doc_title"],
                        "doc_id": record["doc_id"],
                        "score": final_score,
                        "max_sim": max_similarity,
                        "avg_sim": avg_similarity
                    })
                
                # Sort by final score
                scored_chunks.sort(key=lambda x: x["score"], reverse=True)
                top_results = scored_chunks[:final_k]
                
                print(f"\nStage 2: Top {len(top_results)} after re-ranking:")
                for i, chunk in enumerate(top_results, 1):
                    print(f"  {i}. {chunk['doc_title'][:40]} (score: {chunk['score']:.3f})")
                
                if not top_results:
                    return (["No relevant documents found."], [])
                
                # Format results with contribution tracking
                context_chunks = []
                doc_contributions = {}  # Track how much each document contributed
                
                for chunk in top_results:
                    context_chunks.append(chunk["content"])
                    
                    # Track contribution: score + content length
                    doc_id = chunk["doc_id"]
                    doc_title = chunk["doc_title"]
                    chunk_contribution = chunk["score"] * len(chunk["content"])
                    
                    if doc_id not in doc_contributions:
                        doc_contributions[doc_id] = {
                            "title": doc_title,
                            "doc_id": doc_id,
                            "contribution": 0,
                            "chunk_count": 0,
                            "avg_score": 0,
                            "scores": []
                        }
                    
                    doc_contributions[doc_id]["contribution"] += chunk_contribution
                    doc_contributions[doc_id]["chunk_count"] += 1
                    doc_contributions[doc_id]["scores"].append(chunk["score"])
                
                # Calculate average scores and sort by contribution
                for doc_id in doc_contributions:
                    scores = doc_contributions[doc_id]["scores"]
                    doc_contributions[doc_id]["avg_score"] = np.mean(scores)
                
                # Sort sources by contribution (highest first)
                sorted_sources = sorted(
                    doc_contributions.values(),
                    key=lambda x: x["contribution"],
                    reverse=True
                )
                
                print(f"\nSource Contributions:")
                for i, source in enumerate(sorted_sources, 1):
                    print(f"  {i}. {source['title'][:40]}")
                    print(f"     Chunks used: {source['chunk_count']}, Avg score: {source['avg_score']:.3f}")
                
                sources = [
                    {"title": s["title"], "doc_id": s["doc_id"]}
                    for s in sorted_sources
                ]
                
                return (context_chunks, sources)
                
        except Exception as e:
            print(f"Retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return (["Error retrieving context."], [])
    
    def validate_answer(self, question: str, answer: str, context: List[str]) -> Dict:
        """
        Validate if answer actually addresses the question
        """
        validation_prompt = f"""Evaluate this answer:

Question: {question}
Answer: {answer}

Is this answer:
1. Relevant to the question? (yes/no)
2. Based on the provided context? (yes/no)
3. Complete? (yes/no)

Respond with only:
Relevant: [yes/no]
Grounded: [yes/no]
Complete: [yes/no]
Confidence: [low/medium/high]"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=validation_prompt,
                options={'temperature': 0.1, 'num_predict': 50}
            )
            
            if response and 'response' in response:
                text = response['response'].lower()
                return {
                    'relevant': 'relevant: yes' in text,
                    'grounded': 'grounded: yes' in text,
                    'complete': 'complete: yes' in text,
                    'confidence': 'high' if 'confidence: high' in text else 'medium'
                }
        except:
            pass
        
        return {'relevant': True, 'grounded': True, 'complete': True, 'confidence': 'medium'}
    
    def generate_answer(self, question: str, context: List[str], query_analysis: Dict) -> str:
        """Generate answer with improved prompting"""
        if not context or "No relevant" in context[0]:
            return "I couldn't find relevant information in the uploaded documents to answer this question."
        
        context_text = "\n\n".join(context[:5])
        
        # Customize prompt based on query intent
        intent = query_analysis.get('intent', 'unknown')
        
        if 'who' in intent:
            focus = "Focus on identifying the person and their key attributes."
        elif 'what' in intent:
            focus = "Provide a clear definition or explanation."
        elif 'why' in intent:
            focus = "Explain the reasons and causes."
        elif 'how' in intent:
            focus = "Describe the process or method."
        else:
            focus = "Provide a comprehensive answer."
        
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.

Context:
{context_text}

Question: {question}

Instructions:
- {focus}
- Answer based strictly on the context
- Be direct and concise
- If information is incomplete, acknowledge it
- Cite specific details from the context

Answer:"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 250, 'top_p': 0.9}
            )
            
            if response and 'response' in response:
                answer = response['response'].strip()
                
                # Validate answer
                validation = self.validate_answer(question, answer, context)
                
                if not validation['relevant']:
                    return "I found some information but it doesn't directly answer your question. Could you rephrase?"
                
                return answer
            
            return "Could not generate response."
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat(self, question: str) -> Dict:
        """Main chat with advanced understanding"""
        # Step 1: Understand the query
        query_analysis = self.understand_query(question)
        
        # Step 2: Retrieve with re-ranking
        context, sources = self.retrieve_with_reranking(question, query_analysis)
        
        # Step 3: Generate answer
        answer = self.generate_answer(question, context, query_analysis)
        
        return {
            "question": question,
            "answer": answer,
            "context": context[:2],
            "sources": sources
        }