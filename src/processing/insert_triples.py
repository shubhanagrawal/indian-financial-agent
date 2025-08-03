import json
import re
from pathlib import Path
import yaml
from langchain_community.graphs import Neo4jGraph
from collections import defaultdict

# --- Load Configuration ---
config_path = Path(__file__).resolve().parents[2] / 'config/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# --- Graph Database Connection ---
NEO4J_URI = config['neo4j']['uri']
NEO4J_USERNAME = config['neo4j']['username']
NEO4J_PASSWORD = config['neo4j']['password']

try:
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD
    )
    graph.query("RETURN 1")
    print("[SUCCESS] Successfully connected to Neo4j database.")
except Exception as e:
    print(f"[FATAL ERROR] Failed to connect to Neo4j. Error: {e}")
    graph = None

# --- Intelligence Enhancement Mappings ---

# Normalize different ways of saying the same company name
ENTITY_NORMALIZATION_MAP = {
    "Reliance Industries Ltd.": "Reliance Industries Limited",
    "RIL": "Reliance Industries Limited",
    "Jio": "Reliance Jio"
}

# Consolidate similar relationship types into a canonical form
RELATIONSHIP_CONSOLIDATION_MAP = {
    "CEO": "HAS_ROLE",
    "CHAIRMAN": "HAS_ROLE",
    "MANAGING_DIRECTOR": "HAS_ROLE",
    "SUBSIDIARY": "IS_SUBSIDIARY_OF"
}

# Filter out truly generic, low-value relationship verbs
MEANINGLESS_RELATIONS = {
    "IS", "WAS", "ARE", "WERE", "BE", "BEING", "BEEN",
    "HAS", "HAVE", "HAD", "HAVING", "A", "AN", "THE", "OF", "IN", "ON", 
    "AT", "TO", "FOR", "WITH", "ABOUT", "AS", "INTO", "LIKE", "THROUGH",
    "AFTER", "OVER", "BETWEEN", "OUT", "AGAINST", "DURING", "WITHOUT",
    "UNDER", "AROUND", "AMONG", "RELATED_TO"
}


def insert_triples_to_graph():
    """
    Reads the intermediate JSON file, cleans, normalizes, enriches,
    and batch-inserts the triples into Neo4j.
    """
    if not graph:
        print("Graph connection not available. Aborting.")
        return

    company_name = config['processing']['company_name']
    project_root = Path(__file__).resolve().parents[2]
    input_file = project_root / f'data/processed/{company_name}_extracted_triples.json'

    if not input_file.exists():
        print(f"[ERROR] Intermediate triples file not found: {input_file}")
        return

    print(f"--- Starting Intelligent Graph Insertion for {company_name} ---")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        all_triples = json.load(f)

    print(f"Found {len(all_triples)} raw triples to process.")

    print("Clearing old graph data for this company...")
    graph.query(f"MATCH (n {{company: '{company_name}'}}) DETACH DELETE n")

    # --- Data Cleansing and Enrichment Pipeline ---
    
    high_quality_triples = []
    for subject, relationship, obj in all_triples:
        # 1. Normalize Entities
        subject = ENTITY_NORMALIZATION_MAP.get(subject.strip(), subject.strip())
        obj = ENTITY_NORMALIZATION_MAP.get(obj.strip(), obj.strip())

        # 2. Sanitize and Consolidate Relationship
        relationship_clean = relationship.upper().replace(" ", "_")
        relationship_clean = re.sub(r'[^a-zA-Z0-9_]', '', relationship_clean)
        relationship_clean = RELATIONSHIP_CONSOLIDATION_MAP.get(relationship_clean, relationship_clean)

        # 3. Filter out meaningless relationships
        if relationship_clean in MEANINGLESS_RELATIONS or len(relationship_clean) < 3:
            continue

        # 4. Final validation
        if not subject or not obj or not relationship_clean:
            continue
            
        high_quality_triples.append((subject, relationship_clean, obj))

    print(f"Filtered down to {len(high_quality_triples)} high-quality triples.")

    # --- Efficient Batch Insertion ---
    
    # Group triples by relationship type for efficient batching
    rel_groups = defaultdict(list)
    for s, r, o in high_quality_triples:
        rel_groups[r].append({"subject": s, "object": o})

    added_count = 0
    for rel_type, items in rel_groups.items():
        # Construct a single, powerful query for each batch of relationships
        query = """
        UNWIND $items AS item
        MERGE (s:Entity {name: item.subject, company: $company})
        MERGE (o:Entity {name: item.object, company: $company})
        MERGE (s)-[r:""" + rel_type + """]->(o)
        """
        try:
            graph.query(query, params={"items": items, "company": company_name})
            added_count += len(items)
            print(f"Successfully added {len(items)} relationships of type '{rel_type}'")
        except Exception as e:
            print(f"[WARNING] Could not insert batch for relationship '{rel_type}'. Error: {e}")

    print(f"\n--- Graph Insertion Complete ---")
    print(f"Successfully added {added_count} intelligent relationships to the graph.")

if __name__ == '__main__':
    insert_triples_to_graph()
