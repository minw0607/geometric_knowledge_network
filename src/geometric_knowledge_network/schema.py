from enum import Enum


class NodeType(str, Enum):
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    CONCEPT = "Concept"


class EdgeType(str, Enum):
    CONTAINS = "CONTAINS"
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"
