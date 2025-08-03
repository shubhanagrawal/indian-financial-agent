import os
import re
from pathlib import Path
import traceback
import hashlib
import pickle
from typing import List, Tuple, Dict, Optional
import concurrent.futures
from collections import defaultdict
import logging
from datetime import datetime

from langchain_community.graphs import Neo4jGraph
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
from neo4j.exceptions import GqlError, ServiceUnavailable, AuthError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_builder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Graph Database Connection ---
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

def initialize_neo4j_connection(max_retries: int = 3) -> Optional[Neo4jGraph]:
    """Initialize Neo4j connection with error handling and retries"""
    for attempt in range(max_retries):
        try:
            graph = Neo4jGraph(
                url=NEO4J_URI, 
                username=NEO4J_USERNAME, 
                password=NEO4J_PASSWORD,
                timeout=30
            )
            # Test connection
            graph.query("RETURN 1 as test", timeout=10)
            logger.info(f"Successfully connected to Neo4j database on attempt {attempt + 1}")
            return graph
        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            return None
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            if attempt == max_retries - 1:
                return None
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All connection attempts failed")
                return None
    return None

# Initialize graph connection
graph = initialize_neo4j_connection()

def initialize_llm_pipeline():
    """Initialize LLM pipeline with error handling"""
    try:
        model_name = "google/flan-t5-large"
        logger.info(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_length=512,
            do_sample=False,
            temperature=0.1
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("LLM pipeline initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM pipeline: {e}")
        return None

# Initialize LLM
llm = initialize_llm_pipeline()

# --- Prompt for Knowledge Graph Extraction ---
GRAPH_EXTRACTION_TEMPLATE = """
Extract structured information from the text below as relationship triples.

RULES:
1. Format: (Subject, Relationship, Object) - one per line
2. Focus on: companies, people, financial metrics, products, business relationships
3. Use clear, specific relationships like: OWNS, SUBSIDIARY_OF, COMPETES_WITH, HAS_REVENUE
4. Avoid generic words like: is, has, the, a, an

Example:
Text: "Reliance Jio reported revenue of 15000 crores. It is a subsidiary of Reliance Industries."
Output:
(Reliance Jio, HAS_REVENUE, 15000 crores)
(Reliance Jio, SUBSIDIARY_OF, Reliance Industries)

Text: {text}
Output:
"""

graph_extraction_prompt = PromptTemplate(
    template=GRAPH_EXTRACTION_TEMPLATE,
    input_variables=["text"],
)

class ProcessingCache:
    """Handles caching of processed chunks to avoid reprocessing"""
    
    def __init__(self, cache_file: str = "kg_processing_cache.pkl"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.hits = 0
        self.misses = 0
    
    def _load_cache(self) -> dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    logger.info(f"Loaded cache with {len(cache_data)} entries")
                    return cache_data
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
        return {}
    
    def save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved cache with {len(self.cache)} entries. Hits: {self.hits}, Misses: {self.misses}")
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def get_chunk_hash(self, chunk: str) -> str:
        return hashlib.md5(chunk.encode('utf-8')).hexdigest()
    
    def get_cached_result(self, chunk: str) -> Optional[List[Tuple[str, str, str]]]:
        chunk_hash = self.get_chunk_hash(chunk)
        if chunk_hash in self.cache:
            self.hits += 1
            return self.cache[chunk_hash]
        self.misses += 1
        return None
    
    def cache_result(self, chunk: str, result: List[Tuple[str, str, str]]):
        chunk_hash = self.get_chunk_hash(chunk)
        self.cache[chunk_hash] = result

# Initialize cache
cache = ProcessingCache()

def smart_chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    """Create chunks respecting sentence boundaries and token limits"""
    if not text or not text.strip():
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return [text.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Rough token estimation (1 token ≈ 4 characters for English)
        current_tokens = len(current_chunk) // 4
        sentence_tokens = len(sentence) // 4
        
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text.strip()]

def normalize_relationship(relationship: str) -> str:
    """Comprehensive relationship normalization for Neo4j"""
    if not relationship or not relationship.strip():
        return "RELATED_TO"
    
    # Clean the relationship
    rel = str(relationship).strip()
    
    # Remove problematic characters, keep alphanumeric, spaces, hyphens, underscores
    rel = re.sub(r'[^\w\s\-]', '', rel)
    rel = rel.replace(' ', '_').replace('-', '_')
    rel = rel.upper()
    
    # Handle multiple underscores
    rel = re.sub(r'_+', '_', rel)
    rel = rel.strip('_')
    
    if not rel:
        return "RELATED_TO"
    
    # Handle numbers at start
    if rel[0].isdigit():
        rel = "R_" + rel
    
    # Handle Neo4j reserved keywords
    NEO4J_RESERVED = {
        'AND', 'OR', 'XOR', 'NOT', 'NULL', 'TRUE', 'FALSE',
        'CONSTRAINT', 'CREATE', 'DELETE', 'DROP', 'EXISTS',
        'INDEX', 'MATCH', 'MERGE', 'REMOVE', 'RETURN', 'SET',
        'UNION', 'UNWIND', 'WHERE', 'WITH', 'CALL', 'YIELD',
        'DISTINCT', 'ORDER', 'BY', 'LIMIT', 'SKIP', 'CASE',
        'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'ASC', 'DESC'
    }
    
    if rel in NEO4J_RESERVED:
        rel = "REL_" + rel
    
    # Length limit
    if len(rel) > 100:
        rel = rel[:97] + "___"
    
    # Final validation
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', rel):
        logger.debug(f"Invalid relationship name '{rel}', using default")
        return "RELATED_TO"
    
    return rel

def clean_entity_name(name: str) -> Optional[str]:
    """Clean entity names for Neo4j compatibility"""
    if not name or not name.strip():
        return None
    
    # Clean the name
    name = str(name).strip()
    
    # Remove excessive whitespace and newlines
    name = ' '.join(name.split())
    
    # Remove problematic characters but keep meaningful punctuation
    name = re.sub(r'[^\w\s\-\.\,\(\)\&\$\%\#\@]', '', name)
    name = name.strip()
    
    # Length validation
    if len(name) > 200:
        name = name[:197] + "..."
    
    # Minimum length check
    if len(name) < 2:
        return None
        
    return name

def validate_and_clean_triple(subject: str, relation: str, obj: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Validate and clean a triple before database insertion"""
    
    # Clean all components
    clean_subject = clean_entity_name(subject)
    clean_object = clean_entity_name(obj)
    clean_relation = normalize_relationship(relation)
    
    # Basic validation
    if not clean_subject or not clean_object or not clean_relation:
        return None, None, None
    
    # Avoid meaningless self-references
    if clean_subject.lower() == clean_object.lower():
        if clean_relation not in ['IS_A', 'TYPE_OF', 'INSTANCE_OF', 'SAME_AS']:
            return None, None, None
    
    # Filter meaningless relationships
    MEANINGLESS_RELATIONS = {
        'IS', 'HAS', 'THE', 'A', 'AN', 'OF', 'IN', 'ON', 'AT', 'TO', 'FOR',
        'AND', 'BUT', 'OR', 'SO', 'YET', 'BECAUSE', 'WITH', 'BY', 'FROM'
    }
    
    if clean_relation in MEANINGLESS_RELATIONS:
        return None, None, None
    
    return clean_subject, clean_relation, clean_object

def extract_triples_from_text(text: str) -> List[Tuple[str, str, str]]:
    """Extract triples from LLM output using multiple regex patterns"""
    if not text:
        return []
    
    # Multiple patterns to catch different formats
    patterns = [
        r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)',  # (subject, relation, object)
        r'\[\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\]]+?)\s*\]',  # [subject, relation, object]
        r'^([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.+?)$',              # subject, relation, object
    ]
    
    triples = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        if matches:
            triples.extend(matches)
            break  # Use first successful pattern
    
    return triples

def extract_triples_from_chunk(chunk: str, chunk_id: int) -> List[Tuple[str, str, str]]:
    """Extract triples from a chunk with caching and error handling"""
    
    # Check cache first
    cached_result = cache.get_cached_result(chunk)
    if cached_result is not None:
        logger.debug(f"Cache hit for chunk {chunk_id}")
        return cached_result
    
    if not llm:
        logger.error("LLM not available")
        return []
    
    try:
        # Use the newer invoke method
        chain = LLMChain(llm=llm, prompt=graph_extraction_prompt)
        result = chain.invoke({"text": chunk})
        extracted_text = result.get("text", "")
        
        logger.debug(f"LLM output for chunk {chunk_id}: {extracted_text[:200]}...")
        
        # Extract triples using regex
        raw_triples = extract_triples_from_text(extracted_text)
        
        # Clean and validate triples
        cleaned_triples = []
        for s, r, o in raw_triples:
            subject, relation, obj = validate_and_clean_triple(s, r, o)
            if subject and relation and obj:
                cleaned_triples.append((subject, relation, obj))
        
        # Cache the result
        cache.cache_result(chunk, cleaned_triples)
        
        logger.debug(f"Extracted {len(cleaned_triples)} valid triples from chunk {chunk_id}")
        return cleaned_triples
        
    except Exception as e:
        logger.error(f"Failed to process chunk {chunk_id}: {e}")
        return []

def execute_batch_query(query: str, params: dict, max_retries: int = 3) -> bool:
    """Execute a database query with retries"""
    if not graph:
        return False
        
    for attempt in range(max_retries):
        try:
            graph.query(query, params=params, timeout=30)
            return True
        except Exception as e:
            logger.warning(f"Query attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Query failed after {max_retries} attempts: {e}")
                return False
    return False

def batch_insert_triples(triples_batch: List[Tuple[str, str, str]], company_name: str, batch_size: int = 25) -> int:
    """Insert triples in batches with comprehensive error handling"""
    if not triples_batch or not graph:
        return 0
    
    successful_inserts = 0
    failed_triples = []
    
    # Process in smaller batches
    for i in range(0, len(triples_batch), batch_size):
        batch = triples_batch[i:i + batch_size]
        
        # Group by relationship type for efficient queries
        relationship_groups = defaultdict(list)
        for subject, relationship, obj in batch:
            # Final validation
            clean_subject, clean_relation, clean_object = validate_and_clean_triple(subject, relationship, obj)
            if clean_subject and clean_relation and clean_object:
                relationship_groups[clean_relation].append((clean_subject, clean_object))
        
        # Execute queries for each relationship type
        for rel_type, pairs in relationship_groups.items():
            # Validate relationship name one final time
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', rel_type):
                logger.warning(f"Skipping invalid relationship type: {rel_type}")
                continue
            
            try:
                # Build parameterized query
                query_parts = []
                params = {'company': company_name}
                
                for idx, (subject, obj) in enumerate(pairs):
                    query_parts.append(f"""
                        MERGE (s{idx}:Entity {{name: $subject{idx}, company: $company}})
                        MERGE (o{idx}:Entity {{name: $object{idx}, company: $company}})
                        MERGE (s{idx})-[r{idx}:{rel_type}]->(o{idx})
                    """)
                    params[f'subject{idx}'] = subject
                    params[f'object{idx}'] = obj
                
                if query_parts:
                    full_query = "\n".join(query_parts)
                    success = execute_batch_query(full_query, params)
                    
                    if success:
                        successful_inserts += len(pairs)
                        logger.info(f"Successfully inserted {len(pairs)} {rel_type} relationships")
                    else:
                        # Try individual inserts as fallback
                        for subject, obj in pairs:
                            individual_success = execute_individual_insert(subject, rel_type, obj, company_name)
                            if individual_success:
                                successful_inserts += 1
                            else:
                                failed_triples.append((subject, rel_type, obj))
                
            except Exception as e:
                logger.error(f"Batch insert failed for {rel_type}: {e}")
                # Fallback to individual inserts
                for subject, obj in pairs:
                    individual_success = execute_individual_insert(subject, rel_type, obj, company_name)
                    if individual_success:
                        successful_inserts += 1
                    else:
                        failed_triples.append((subject, rel_type, obj))
    
    if failed_triples:
        logger.warning(f"{len(failed_triples)} triples failed to insert")
        # Log failed triples to file
        try:
            with open(f"failed_triples_{company_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", "w") as f:
                for triple in failed_triples:
                    f.write(f"{triple[0]} | {triple[1]} | {triple[2]}\n")
        except Exception as e:
            logger.error(f"Could not write failed triples log: {e}")
    
    return successful_inserts

def execute_individual_insert(subject: str, relationship: str, obj: str, company_name: str) -> bool:
    """Insert a single triple with error handling"""
    try:
        query = f"""
            MERGE (s:Entity {{name: $subject, company: $company}})
            MERGE (o:Entity {{name: $object, company: $company}})
            CREATE (s)-[r:{relationship}]->(o)
        """
        return execute_batch_query(query, {
            'subject': subject,
            'object': obj,
            'company': company_name
        })
    except Exception as e:
        logger.debug(f"Individual insert failed: {e}")
        return False

def process_file_optimized(txt_path: Path, company_name: str) -> int:
    """Process a single file with comprehensive error handling"""
    logger.info(f"Processing file: {txt_path.name}")
    
    try:
        # Read file with encoding handling
        text_content = ""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    text_content = f.read()
                logger.debug(f"Successfully read {txt_path.name} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if not text_content:
            logger.error(f"Could not read {txt_path.name} with any encoding")
            return 0
        
        # Smart chunking
        chunks = smart_chunk_text(text_content, max_tokens=400)
        
        if not chunks:
            logger.warning(f"No valid chunks found in {txt_path.name}")
            return 0
        
        logger.info(f"Created {len(chunks)} chunks for {txt_path.name}")
        
        # Process chunks with limited parallelism
        all_triples = []
        max_workers = min(3, len(chunks))  # Limit concurrent LLM calls
        
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {
                    executor.submit(extract_triples_from_chunk, chunk, i): i 
                    for i, chunk in enumerate(chunks)
                }
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        triples = future.result(timeout=60)  # 60 second timeout
                        all_triples.extend(triples)
                        logger.info(f"Processed chunk {chunk_id + 1}/{len(chunks)} from {txt_path.name} - found {len(triples)} triples")
                    except Exception as e:
                        logger.error(f"Chunk {chunk_id} processing failed: {e}")
        else:
            # Sequential processing for small chunk counts
            for i, chunk in enumerate(chunks):
                triples = extract_triples_from_chunk(chunk, i)
                all_triples.extend(triples)
                logger.info(f"Processed chunk {i + 1}/{len(chunks)} from {txt_path.name} - found {len(triples)} triples")
        
        # Deduplicate triples
        unique_triples = list(set(all_triples))
        duplicate_count = len(all_triples) - len(unique_triples)
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate triples from {txt_path.name}")
        
        # Insert to database
        if unique_triples:
            inserted_count = batch_insert_triples(unique_triples, company_name)
            logger.info(f"Successfully inserted {inserted_count}/{len(unique_triples)} relationships from {txt_path.name}")
            return inserted_count
        else:
            logger.warning(f"No valid triples found in {txt_path.name}")
            return 0
            
    except Exception as e:
        logger.error(f"Critical error processing {txt_path.name}: {e}")
        logger.debug(traceback.format_exc())
        return 0

def create_database_indexes(company_name: str):
    """Create database indexes for better performance"""
    if not graph:
        return
        
    indexes = [
        "CREATE INDEX entity_name_company IF NOT EXISTS FOR (e:Entity) ON (e.name, e.company)",
        "CREATE INDEX entity_company IF NOT EXISTS FOR (e:Entity) ON (e.company)",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
    ]
    
    for index_query in indexes:
        try:
            graph.query(index_query)
            logger.info("Created database index")
        except Exception as e:
            logger.debug(f"Index creation skipped (may already exist): {e}")

def get_database_statistics(company_name: str) -> dict:
    """Get comprehensive database statistics"""
    if not graph:
        return {}
    
    try:
        stats = {}
        
        # Entity count
        result = graph.query(f"MATCH (n:Entity {{company: '{company_name}'}}) RETURN count(n) as count")
        stats['entities'] = result[0]['count'] if result else 0
        
        # Relationship count
        result = graph.query(f"MATCH (:Entity {{company: '{company_name}'}})-[r]->(:Entity {{company: '{company_name}'}}) RETURN count(r) as count")
        stats['relationships'] = result[0]['count'] if result else 0
        
        # Relationship types
        result = graph.query(f"""
            MATCH (:Entity {{company: '{company_name}'}})-[r]->(:Entity {{company: '{company_name}'}}) 
            RETURN DISTINCT type(r) as rel_type, count(r) as count 
            ORDER BY count DESC 
            LIMIT 10
        """)
        stats['relationship_types'] = {r['rel_type']: r['count'] for r in result} if result else {}
        
        # Sample entities
        result = graph.query(f"MATCH (n:Entity {{company: '{company_name}'}}) RETURN n.name as name LIMIT 10")
        stats['sample_entities'] = [r['name'] for r in result] if result else []
        
        # Sample relationships
        result = graph.query(f"""
            MATCH (a:Entity {{company: '{company_name}'}})-[r]->(b:Entity {{company: '{company_name}'}}) 
            RETURN a.name as subject, type(r) as relationship, b.name as object 
            LIMIT 10
        """)
        stats['sample_relationships'] = [
            f"{r['subject']} --{r['relationship']}--> {r['object']}" 
            for r in result
        ] if result else []
        
        return stats
        
    except Exception as e:
        logger.error(f"Could not retrieve database statistics: {e}")
        return {}

def build_knowledge_graph(company_name: str):
    """Main function to build knowledge graph with comprehensive error handling"""
    
    if not graph:
        logger.error("Graph connection not available. Aborting.")
        return
    
    if not llm:
        logger.error("LLM not available. Aborting.")
        return
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / f'data/processed/{company_name}'
    
    if not processed_data_dir.exists():
        logger.error(f"Data directory not found: {processed_data_dir}")
        return
    
    logger.info(f"Starting Knowledge Graph construction for {company_name}")
    start_time = datetime.now()
    
    # Clear old data
    try:
        logger.info("Clearing old graph data...")
        result = graph.query(f"MATCH (n {{company: '{company_name}'}}) DETACH DELETE n")
        logger.info("Old graph data cleared successfully")
    except Exception as e:
        logger.error(f"Could not clear old data: {e}")
        return
    
    # Create indexes
    create_database_indexes(company_name)
    
    # Process directories
    dirs_to_process = ["annual_reports", "quarterly_reports", "earnings_transcripts"]
    total_relationships = 0
    total_files = 0
    
    for subdir in dirs_to_process:
        subdir_path = processed_data_dir / subdir
        if not subdir_path.exists():
            logger.warning(f"Directory not found, skipping: {subdir_path}")
            continue
            
        logger.info(f"Processing directory: {subdir}")
        txt_files = list(subdir_path.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"No .txt files found in {subdir_path}")
            continue
        
        logger.info(f"Found {len(txt_files)} files in {subdir}")
        
        # Process files
        for txt_path in txt_files:
            try:
                relationships_added = process_file_optimized(txt_path, company_name)
                total_relationships += relationships_added
                total_files += 1
                logger.info(f"File {txt_path.name}: {relationships_added} relationships added")
            except Exception as e:
                logger.error(f"Failed to process {txt_path.name}: {e}")
    
    # Save cache
    cache.save_cache()
    
    # Final statistics
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    logger.info(f"Knowledge Graph construction completed!")
    logger.info(f"Processing time: {processing_time}")
    logger.info(f"Files processed: {total_files}")
    logger.info(f"Total relationships added: {total_relationships}")
    
    # Database statistics
    stats = get_database_statistics(company_name)
    if stats:
        logger.info(f"Final database statistics:")
        logger.info(f"  Entities: {stats.get('entities', 0)}")
        logger.info(f"  Relationships: {stats.get('relationships', 0)}")
        
        if stats.get('relationship_types'):
            logger.info("  Top relationship types:")
            for rel_type, count in list(stats['relationship_types'].items())[:5]:
                logger.info(f"    {rel_type}: {count}")
        
        if stats.get('sample_relationships'):
            logger.info("  Sample relationships:")
            for rel in stats['sample_relationships'][:3]:
                logger.info(f"    {rel}")
    
    # Visualization query suggestion
    if stats.get('relationships', 0) > 0:
        logger.info("\nTo visualize in Neo4j Browser, try this query:")
        logger.info(f"MATCH (a:Entity {{company: '{company_name}'}})-[r]->(b:Entity {{company: '{company_name}'}}) RETURN a, r, b LIMIT 50")
    else:
        logger.warning("No relationships found in database. Check the logs for processing issues.")

if __name__ == '__main__':
    try:
        target_company = "RELIANCE_INDUSTRIES"
        build_knowledge_graph(target_company)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        # Cleanup
        if 'cache' in locals():
            cache.save_cache()
        logger.info("Script execution completed")