# src/knowledge_graph_builder.py
from typing import Dict, Any

class KnowledgeGraphBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        print("STUB: KnowledgeGraphBuilder initialized.")

    def process_chunk(self, chunk_data: Dict[str, Any]):
        """
        Processes a single audio chunk to extract and link entities.
        STUB: Does nothing.
        """
        chunk_id = chunk_data.get("chunk_id")
        print(f"STUB: Processing chunk {chunk_id} for KG.")
        pass