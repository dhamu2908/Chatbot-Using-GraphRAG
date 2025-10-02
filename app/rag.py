import ollama
import os
from typing import List, Dict, Tuple, Optional
from database import Neo4jConnection
import hashlib
import re
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading

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
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.embedding_cache = {}  # Cache embeddings
        self.cache_lock = threading.Lock()
    
    def _ensure_embedding_model(self):
        """Ensure embedding model is available"""
        try:
            ollama.embeddings(model=self.embedding_model, prompt="test")
        except:
            print(f"Pulling embedding model...")
            os.system(f"ollama pull {self.embedding_model}")
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def fuzzy_match(self, query_term: str, keyword: str, threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Check if query_term fuzzy matches keyword
        Returns (is_match, similarity_score)
        """
        query_lower = query_term.lower()
        keyword_lower = keyword.lower()
        
        # Exact match
        if query_lower == keyword_lower:
            return (True, 1.0)
        
        # Substring match
        if query_lower in keyword_lower or keyword_lower in query_lower:
            return (True, 0.9)
        
        # Calculate similarity based on edit distance
        max_len = max(len(query_lower), len(keyword_lower))
        if max_len == 0:
            return (False, 0.0)
        
        distance = self.levenshtein_distance(query_lower, keyword_lower)
        similarity = 1 - (distance / max_len)
        
        # Match if similarity is above threshold
        is_match = similarity >= threshold
        return (is_match, similarity)
    
    def find_fuzzy_matches(self, query_terms: List[str], all_keywords: List[str], 
                          threshold: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find fuzzy matches for all query terms against all keywords
        Returns dict mapping query_term -> [(matched_keyword, similarity_score), ...]
        """
        fuzzy_matches = {}
        
        for query_term in query_terms:
            if len(query_term) < 3:  # Skip very short terms
                continue
            
            matches = []
            for keyword in all_keywords:
                is_match, similarity = self.fuzzy_match(query_term, keyword, threshold)
                if is_match:
                    matches.append((keyword, similarity))
            
            # Sort by similarity score
            matches.sort(key=lambda x: x[1], reverse=True)
            if matches:
                fuzzy_matches[query_term] = matches[:5]  # Keep top 5 matches per term
        
        return fuzzy_matches
    
    def normalize_entity(self, entity: str) -> str:
        """Normalize entity names for better matching"""
        normalized = entity.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[.]', '', normalized)
        
        prefixes = ['dr', 'mr', 'mrs', 'ms', 'prof', 'sir']
        for prefix in prefixes:
            if normalized.startswith(prefix + ' '):
                normalized = normalized[len(prefix)+1:]
        
        return normalized.strip()
    
    def extract_entity_variations(self, entity: str) -> List[str]:
        """Generate variations of an entity name"""
        variations = set()
        normalized = self.normalize_entity(entity)
        variations.add(normalized)
        
        words = normalized.split()
        if len(words) > 1:
            no_initials = ' '.join([w for w in words if len(w) > 1])
            if no_initials:
                variations.add(no_initials)
        
        variations.add(normalized.replace(' ', ''))
        return list(variations)
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate semantic embedding with caching"""
        # Create cache key
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
        
        # Check cache
        with self.cache_lock:
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text.strip()[:1000]
            )
            embedding = response['embedding']
            
            # Store in cache
            with self.cache_lock:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
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
        """Robust query understanding with regex-first approach"""
        print(f"\n{'='*60}")
        print(f"UNDERSTANDING QUERY: {question}")
        print(f"{'='*60}")
        
        # DETERMINISTIC ENTITY EXTRACTION (doesn't rely on LLM)
        direct_entities = []
        
        # 1. Extract capitalized multi-word names (e.g., "Shubman Gill", "Yuzi Chahal")
        full_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', question)
        direct_entities.extend(full_names)
        print(f"Full names found: {full_names}")
        
        # 2. Extract single capitalized words (e.g., "Shubman", "Gill", "Chahal")
        single_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', question)
        direct_entities.extend(single_caps)
        print(f"Single capitalized words: {single_caps}")
        
        # 3. Handle lowercase names (casual typing like "shubman gill")
        words = question.lower().split()
        for i in range(len(words)):
            # Check for two consecutive words that could be a name
            if i < len(words) - 1:
                word1, word2 = words[i], words[i+1]
                # Skip common words
                if len(word1) > 2 and len(word2) > 2 and word1 not in {'the', 'and', 'about', 'what', 'who', 'how', 'when', 'where'}:
                    potential_name = f"{word1.capitalize()} {word2.capitalize()}"
                    direct_entities.append(potential_name)
                    print(f"Potential name from lowercase: {potential_name}")
            
            # Also add individual words as potential entities
            if len(words[i]) > 3 and words[i] not in {'the', 'and', 'about', 'what', 'who', 'how', 'when', 'where', 'this', 'that', 'with', 'from'}:
                direct_entities.append(words[i].capitalize())
        
        # 4. Extract numbers (for queries about amounts, years, etc.)
        numbers = re.findall(r'\d+', question)
        direct_entities.extend(numbers)
        
        print(f"\nAll direct entities extracted: {direct_entities}")
        
        # Generate variations for all entities
        entity_variations = []
        for entity in direct_entities:
            if entity and len(entity) > 1:
                variations = self.extract_entity_variations(entity)
                entity_variations.extend(variations)
                
                # For multi-word names, also add each word separately
                if ' ' in entity:
                    words = entity.split()
                    for word in words:
                        if len(word) > 2:
                            entity_variations.extend(self.extract_entity_variations(word))
        
        # Remove duplicates
        entity_variations = list(dict.fromkeys(entity_variations))
        
        # Extract keywords from question
        keyword_variations = []
        important_words = re.findall(r'\b[a-z]{3,}\b', question.lower())
        stop_words = {
            'the', 'and', 'about', 'what', 'who', 'how', 'when', 'where', 'why',
            'tell', 'something', 'anything', 'this', 'that', 'with', 'from', 'have'
        }
        for word in important_words:
            if word not in stop_words:
                keyword_variations.extend(self.extract_entity_variations(word))
        
        keyword_variations = list(dict.fromkeys(keyword_variations))
        
        # Detect question type
        question_lower = question.lower()
        if any(w in question_lower for w in ['who is', 'who was', 'who']):
            intent = 'who'
        elif any(w in question_lower for w in ['what is', 'what was', 'what happened', 'what']):
            intent = 'what'
        elif any(w in question_lower for w in ['how much', 'how many']):
            intent = 'how_much'
        elif any(w in question_lower for w in ['why', 'reason']):
            intent = 'why'
        elif any(w in question_lower for w in ['when', 'time', 'date']):
            intent = 'when'
        else:
            intent = 'unknown'
        
        result = {
            'original': question,
            'intent': intent,
            'entities': entity_variations[:25],
            'expanded': [question],
            'keywords': keyword_variations[:25]
        }
        
        print(f"\n{'='*60}")
        print(f"FINAL QUERY ANALYSIS:")
        print(f"  Intent: {result['intent']}")
        print(f"  Entities ({len(result['entities'])}): {result['entities'][:10]}")
        print(f"  Keywords ({len(result['keywords'])}): {result['keywords'][:10]}")
        print(f"{'='*60}\n")
        
        return result
    
    def extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction"""
        entities = []
        proper = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend([p for p in proper])
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        entities.extend([a for a in acronyms])
        initials = re.findall(r'\b[A-Z]\.(?:[A-Z]\.)*[A-Z][a-z]+', text)
        entities.extend(initials)
        return list(set(entities))[:5]
    
    def chunk_text(self, text: str, max_chunk_size: int = 150) -> List[str]:
        """Optimized intelligent chunking"""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current = []
        word_count = 0
        
        for sent in sentences:
            words = len(sent.split())
            if word_count + words > max_chunk_size and current:
                chunks.append('. '.join(current) + '.')
                current = [current[-1]] if current else []
                word_count = len(current[0].split()) if current else 0
            current.append(sent)
            word_count += words
        
        if current:
            chunks.append('. '.join(current) + '.')
        
        return chunks if chunks else [text]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction with better name recognition"""
        keywords = set()
        
        # Extract full names (2-3 words starting with capitals)
        full_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
        for name in full_names:
            # Add full name and variations
            keywords.update(self.extract_entity_variations(name))
            # Also add individual name parts
            name_parts = name.split()
            for part in name_parts:
                if len(part) > 2:
                    keywords.update(self.extract_entity_variations(part))
        
        # Extract names with initials (Yuzi Chahal, B.R.Ambedkar style)
        initials_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z]\.?)*(?:\s+[A-Z][a-z]+)+\b', text)
        for name in initials_names:
            keywords.update(self.extract_entity_variations(name))
        
        # Extract single capitalized words (last names, places)
        single_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in single_caps:
            keywords.update(self.extract_entity_variations(word))
        
        # Extract acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        for a in acronyms:
            keywords.update(self.extract_entity_variations(a))
        
        # Extract important words (non-stopwords, frequency-based)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
            'what', 'who', 'when', 'where', 'why', 'how', 'this', 'that', 'these',
            'those', 'have', 'has', 'had', 'will', 'would', 'could', 'should'
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = Counter(w for w in words if w not in stop_words)
        keywords.update([w for w, c in word_freq.most_common(10)])
        
        # Also extract numbers and special terms (amounts, years, etc.)
        numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
        keywords.update(numbers[:5])  # Add important numbers
        
        return list(keywords)[:30]  # Increased from 20 to 30
    
    def process_chunk(self, chunk: str, chunk_id: str, doc_id: str):
        """Process a single chunk (for parallel execution)"""
        try:
            embedding = self.get_embedding(chunk)
            if not embedding:
                return None
            
            keywords = self.extract_keywords(chunk)
            
            return {
                'chunk_id': chunk_id,
                'content': chunk,
                'doc_id': doc_id,
                'embedding': embedding,
                'keywords': keywords
            }
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            return None
    
    def store_document(self, title: str, content: str):
        """Store document with parallel chunk processing"""
        if not hasattr(self.neo4j, 'driver') or not self.neo4j.driver:
            print("Neo4j not available")
            return
        
        doc_id = hashlib.md5(f"{title}{content}".encode()).hexdigest()[:16]
        chunks = self.chunk_text(content)
        
        print(f"\nStoring: {title} ({len(chunks)} chunks)")
        
        try:
            with self.neo4j.driver.session() as session:
                # Create document node
                session.run(
                    "MERGE (d:Document {id: $id, title: $title})",
                    id=doc_id, title=title
                )
                
                # Process chunks in parallel
                chunk_data_list = []
                futures = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    future = self.executor.submit(
                        self.process_chunk, 
                        chunk, 
                        chunk_id, 
                        doc_id
                    )
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    result = future.result()
                    if result:
                        chunk_data_list.append(result)
                
                # Batch insert chunks and relationships
                for chunk_data in chunk_data_list:
                    # Create chunk node
                    session.run(
                        """
                        MERGE (c:Chunk {
                            id: $id, content: $content, document_id: $doc_id,
                            embedding: $embedding
                        })
                        """,
                        id=chunk_data['chunk_id'],
                        content=chunk_data['content'],
                        doc_id=chunk_data['doc_id'],
                        embedding=chunk_data['embedding']
                    )
                    
                    # Create document-chunk relationship
                    session.run(
                        "MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id}) "
                        "MERGE (d)-[:CONTAINS]->(c)",
                        doc_id=chunk_data['doc_id'],
                        chunk_id=chunk_data['chunk_id']
                    )
                    
                    # Create keyword relationships in batch
                    for kw in chunk_data['keywords']:
                        if kw and len(kw) > 1:
                            session.run("MERGE (k:Keyword {name: $keyword})", keyword=kw)
                            session.run(
                                "MATCH (c:Chunk {id: $chunk_id}), (k:Keyword {name: $keyword}) "
                                "MERGE (c)-[:HAS_KEYWORD]->(k)",
                                chunk_id=chunk_data['chunk_id'],
                                keyword=kw
                            )
                
                print(f"  Stored successfully with {len(chunk_data_list)} chunks")
                
        except Exception as e:
            print(f"Error storing document: {e}")
    
    def retrieve_with_reranking(self, question: str, query_analysis: Dict, 
                                initial_k: int = 20, final_k: int = 8) -> Tuple[List[str], List[Dict]]:
        """Enhanced retrieval with fuzzy matching for misspelled queries"""
        if not hasattr(self.neo4j, 'driver') or not self.neo4j.driver:
            return (["Database not available."], [])
        
        print(f"\n{'='*60}")
        print(f"QUERY ANALYSIS:")
        print(f"  Question: {question}")
        print(f"  Intent: {query_analysis['intent']}")
        print(f"  Entities: {query_analysis['entities'][:10]}")
        print(f"  Keywords: {query_analysis['keywords'][:10]}")
        print(f"{'='*60}")
        
        # Get query embedding
        query_embedding = self.get_embedding(question)
        if not query_embedding:
            return (["Failed to generate embeddings."], [])
        
        try:
            with self.neo4j.driver.session() as session:
                all_keywords = list(set(query_analysis['keywords'] + query_analysis['entities']))
                
                print(f"\nSearching with {len(all_keywords)} total keywords: {all_keywords[:15]}")
                
                # STEP 1: Get all available keywords from database
                result = session.run("MATCH (k:Keyword) RETURN k.name as keyword")
                db_keywords = [record['keyword'] for record in result]
                print(f"\nDatabase contains {len(db_keywords)} unique keywords")
                
                # STEP 2: Find fuzzy matches for potentially misspelled query terms
                fuzzy_threshold = 0.7  # 70% similarity required
                fuzzy_matches = self.find_fuzzy_matches(all_keywords, db_keywords, fuzzy_threshold)
                
                # Build expanded keyword list with fuzzy matches
                expanded_keywords = list(all_keywords)  # Start with original keywords
                
                if fuzzy_matches:
                    print(f"\n{'='*60}")
                    print("FUZZY MATCHES FOUND:")
                    for query_term, matches in fuzzy_matches.items():
                        print(f"  '{query_term}' matches:")
                        for matched_kw, score in matches:
                            print(f"    - '{matched_kw}' (similarity: {score:.2f})")
                            if matched_kw not in expanded_keywords:
                                expanded_keywords.append(matched_kw)
                    print(f"{'='*60}\n")
                else:
                    print("\nNo fuzzy matches needed (exact matches found or no close matches)")
                
                print(f"Expanded keyword list: {len(expanded_keywords)} keywords")
                
                # STRATEGY 1: Aggressive keyword matching with fuzzy-matched keywords
                keyword_candidates = []
                if expanded_keywords:
                    result = session.run(
                        """
                        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[:HAS_KEYWORD]->(k:Keyword)
                        WHERE k.name IN $keywords
                        WITH DISTINCT c, d, count(DISTINCT k) as keyword_count
                        RETURN c.id as chunk_id, c.content as content, c.embedding as embedding,
                               d.title as doc_title, d.id as doc_id, keyword_count
                        ORDER BY keyword_count DESC
                        LIMIT 50
                        """,
                        keywords=expanded_keywords[:50]
                    )
                    keyword_candidates = list(result)
                    print(f"KEYWORD SEARCH: Found {len(keyword_candidates)} chunks")
                    for i, cand in enumerate(keyword_candidates[:3]):
                        print(f"  {i+1}. {cand['doc_title'][:50]} - matches: {cand['keyword_count']}")
                
                # STRATEGY 2: Text content search with fuzzy matching
                content_candidates = []
                result = session.run(
                    """
                    MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
                    RETURN c.id as chunk_id, c.content as content, c.embedding as embedding,
                           d.title as doc_title, d.id as doc_id
                    """
                )
                all_chunks = list(result)
                
                # Filter by content matching (including fuzzy matched terms)
                for chunk in all_chunks:
                    content_lower = chunk['content'].lower()
                    content_normalized = self.normalize_entity(chunk['content'])
                    
                    # Check original entities
                    matches = 0
                    for entity in query_analysis['entities'][:15]:
                        if entity.lower() in content_lower or entity in content_normalized:
                            matches += 1
                    
                    # Check fuzzy matched keywords in content
                    for query_term, matched_keywords in fuzzy_matches.items():
                        for matched_kw, score in matched_keywords:
                            if matched_kw.lower() in content_lower:
                                matches += score  # Weight by similarity score
                    
                    if matches > 0:
                        content_candidates.append({
                            'chunk_id': chunk['chunk_id'],
                            'content': chunk['content'],
                            'embedding': chunk['embedding'],
                            'doc_title': chunk['doc_title'],
                            'doc_id': chunk['doc_id'],
                            'content_matches': matches,
                            'keyword_count': 0
                        })
                
                print(f"CONTENT SEARCH: Found {len(content_candidates)} chunks with matching text")
                for i, cand in enumerate(sorted(content_candidates, key=lambda x: x['content_matches'], reverse=True)[:3]):
                    print(f"  {i+1}. {cand['doc_title'][:50]} - text matches: {cand['content_matches']:.2f}")
                
                # STRATEGY 3: Combine all candidates
                candidates_dict = {}
                
                # Add keyword candidates
                for cand in keyword_candidates:
                    candidates_dict[cand['chunk_id']] = {
                        'chunk_id': cand['chunk_id'],
                        'content': cand['content'],
                        'embedding': cand['embedding'],
                        'doc_title': cand['doc_title'],
                        'doc_id': cand['doc_id'],
                        'keyword_count': cand['keyword_count'],
                        'content_matches': 0
                    }
                
                # Add or merge content candidates
                for cand in content_candidates:
                    if cand['chunk_id'] in candidates_dict:
                        candidates_dict[cand['chunk_id']]['content_matches'] = cand['content_matches']
                    else:
                        candidates_dict[cand['chunk_id']] = cand
                
                candidates = list(candidates_dict.values())
                
                # If still no candidates, get top chunks by any means
                if len(candidates) < 5:
                    print(f"WARNING: Only {len(candidates)} candidates, adding more...")
                    result = session.run(
                        """
                        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
                        RETURN c.id as chunk_id, c.content as content, c.embedding as embedding,
                               d.title as doc_title, d.id as doc_id
                        LIMIT 100
                        """
                    )
                    for record in result:
                        if record['chunk_id'] not in candidates_dict:
                            candidates.append({
                                'chunk_id': record['chunk_id'],
                                'content': record['content'],
                                'embedding': record['embedding'],
                                'doc_title': record['doc_title'],
                                'doc_id': record['doc_id'],
                                'keyword_count': 0,
                                'content_matches': 0
                            })
                
                print(f"\nTOTAL CANDIDATES: {len(candidates)}")
                
                # Score all candidates
                scored_chunks = []
                for record in candidates:
                    chunk_embedding = record["embedding"]
                    if not chunk_embedding:
                        continue
                    
                    # Semantic similarity
                    similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                    
                    # Keyword match boost (graph-based)
                    keyword_boost = min(record.get('keyword_count', 0) * 0.15, 0.4)
                    
                    # Content match boost (text-based, including fuzzy matches)
                    content_boost = min(record.get('content_matches', 0) * 0.2, 0.5)
                    
                    # Combined score - prioritize content matches
                    final_score = similarity + keyword_boost + content_boost
                    
                    scored_chunks.append({
                        "chunk_id": record["chunk_id"],
                        "content": record["content"],
                        "doc_title": record["doc_title"],
                        "doc_id": record["doc_id"],
                        "score": final_score,
                        "similarity": similarity,
                        "keyword_count": record.get('keyword_count', 0),
                        "content_matches": record.get('content_matches', 0)
                    })
                
                # Sort by score
                scored_chunks.sort(key=lambda x: x["score"], reverse=True)
                top_results = scored_chunks[:final_k]
                
                print(f"\n{'='*60}")
                print(f"TOP {len(top_results)} RESULTS:")
                for i, chunk in enumerate(top_results, 1):
                    print(f"{i}. {chunk['doc_title'][:40]}")
                    print(f"   Score: {chunk['score']:.3f} (sim: {chunk['similarity']:.3f}, kw: {chunk['keyword_count']}, txt: {chunk['content_matches']:.2f})")
                    print(f"   Preview: {chunk['content'][:100]}...")
                print(f"{'='*60}")
                
                if not top_results:
                    return (["No relevant documents found."], [])
                
                context_chunks = []
                doc_contributions = {}
                
                for chunk in top_results:
                    context_chunks.append(chunk["content"])
                    
                    doc_id = chunk["doc_id"]
                    doc_title = chunk["doc_title"]
                    chunk_contribution = chunk["score"] * len(chunk["content"])
                    
                    if doc_id not in doc_contributions:
                        doc_contributions[doc_id] = {
                            "title": doc_title,
                            "doc_id": doc_id,
                            "contribution": 0,
                            "chunk_count": 0,
                            "scores": []
                        }
                    
                    doc_contributions[doc_id]["contribution"] += chunk_contribution
                    doc_contributions[doc_id]["chunk_count"] += 1
                    doc_contributions[doc_id]["scores"].append(chunk["score"])
                
                sorted_sources = sorted(
                    doc_contributions.values(),
                    key=lambda x: x["contribution"],
                    reverse=True
                )[:5]
                
                print(f"\nSOURCE DOCUMENTS (by contribution):")
                for i, source in enumerate(sorted_sources, 1):
                    avg_score = np.mean(source['scores'])
                    print(f"  {i}. {source['title']}")
                    print(f"     Chunks: {source['chunk_count']}, Avg score: {avg_score:.3f}")
                
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
    
    def generate_answer(self, question: str, context: List[str], query_analysis: Dict) -> str:
        """Generate answer with improved context utilization"""
        if not context or "No relevant" in context[0]:
            return "I couldn't find relevant information in the uploaded documents to answer this question."
        
        # Combine all context
        context_text = "\n\n".join(context[:5])
        intent = query_analysis.get('intent', 'unknown')
        
        # Extract key entities from question for focus
        question_entities = query_analysis.get('entities', [])
        entity_focus = ""
        if question_entities:
            entity_focus = f"Pay special attention to information about: {', '.join(question_entities[:3])}"
        
        if 'who' in intent:
            focus = "Identify the person and provide specific details about them from the context."
        elif 'what' in intent:
            focus = "Provide specific information and details from the context."
        elif 'why' in intent:
            focus = "Explain the reasons using specific details from the context."
        elif 'how' in intent or 'much' in question.lower():
            focus = "Provide specific numbers, amounts, or details from the context."
        else:
            focus = "Provide specific information from the context."
        
        # More directive prompt that forces LLM to read context carefully
        prompt = f"""Based on the following context, answer the question with specific details.

CONTEXT:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
1. {focus}
2. {entity_focus}
3. Quote or reference specific facts, numbers, or details from the context
4. If the exact answer isn't in the context, provide the closest relevant information
5. Do NOT say "the context does not provide" if there is ANY related information
6. Extract and present ANY relevant details you find

ANSWER (be specific and use details from context):"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 250,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            if response and 'response' in response:
                answer = response['response'].strip()
                
                # Check if answer is a refusal when context actually has info
                refusal_phrases = [
                    "does not contain",
                    "does not provide",
                    "no information about",
                    "doesn't mention"
                ]
                
                is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)
                
                if is_refusal and len(context_text) > 100:
                    # Force extraction of relevant information
                    extraction_prompt = f"""The context below contains information. Extract and summarize ANY relevant details related to: {question}

CONTEXT:
{context_text}

Extract specific facts, names, numbers, or details that relate to the question. Focus on what IS mentioned rather than what isn't:"""
                    
                    retry_response = ollama.generate(
                        model=self.model_name,
                        prompt=extraction_prompt,
                        options={'temperature': 0.2, 'num_predict': 200}
                    )
                    
                    if retry_response and 'response' in retry_response:
                        extracted = retry_response['response'].strip()
                        if extracted and len(extracted) > 20:
                            return extracted
                
                return answer
            
            return "Could not generate response."
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat(self, question: str) -> Dict:
        """Main chat with optimized processing"""
        query_analysis = self.understand_query(question)
        context, sources = self.retrieve_with_reranking(question, query_analysis)
        answer = self.generate_answer(question, context, query_analysis)
        
        return {
            "question": question,
            "answer": answer,
            "context": context[:2],
            "sources": sources
        }