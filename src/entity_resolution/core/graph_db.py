"""
Neo4j graph database interface for entity resolution
"""
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

try:
    from neo4j import GraphDatabase, Driver, Session
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    Driver = None
    Session = None

from src.entity_resolution.core.entities import Entity, Relationship, EntityType, RelationshipType
from config.settings import settings


class Neo4jConnection:
    """
    Neo4j database connection and query interface
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j URI (defaults to settings)
            user: Neo4j username (defaults to settings)
            password: Neo4j password (defaults to settings)
            database: Neo4j database name (defaults to settings)
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package is not installed. Install with: pip install neo4j")

        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database

        self.driver: Optional[Driver] = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            print(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result dictionaries
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")

        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def create_entity_node(self, entity: Entity) -> Dict[str, Any]:
        """
        Create or update entity node in graph

        Args:
            entity: Entity object to create

        Returns:
            Created/updated node data
        """
        query = """
        MERGE (e:Entity {entity_id: $entity_id})
        ON CREATE SET
            e.name = $name,
            e.original_name = $original_name,
            e.entity_type = $entity_type,
            e.created_at = datetime(),
            e.confidence = $confidence
        ON MATCH SET
            e.last_updated = datetime(),
            e.confidence = CASE
                WHEN $confidence > e.confidence THEN $confidence
                ELSE e.confidence
            END
        SET
            e.addresses = $addresses,
            e.jurisdiction = $jurisdiction,
            e.identifiers = $identifiers,
            e.phone_numbers = $phone_numbers,
            e.email_addresses = $email_addresses,
            e.websites = $websites,
            e.industry_codes = $industry_codes,
            e.roles = $roles,
            e.data_sources = $data_sources
        RETURN e
        """

        parameters = {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "original_name": entity.original_name,
            "entity_type": entity.entity_type.value,
            "confidence": entity.confidence,
            "addresses": entity.addresses,
            "jurisdiction": entity.jurisdiction,
            "identifiers": entity.identifiers,
            "phone_numbers": entity.phone_numbers,
            "email_addresses": entity.email_addresses,
            "websites": entity.websites,
            "industry_codes": entity.industry_codes,
            "roles": entity.roles,
            "data_sources": entity.data_sources,
        }

        results = self.execute_query(query, parameters)
        return results[0] if results else {}

    def create_relationship_edge(self, relationship: Relationship) -> Dict[str, Any]:
        """
        Create relationship edge between entities

        Args:
            relationship: Relationship object to create

        Returns:
            Created/updated relationship data
        """
        query = """
        MATCH (source:Entity {entity_id: $source_id})
        MATCH (target:Entity {entity_id: $target_id})
        MERGE (source)-[r:RELATIONSHIP {
            relationship_id: $relationship_id
        }]->(target)
        ON CREATE SET
            r.relationship_type = $relationship_type,
            r.created_at = datetime(),
            r.confidence = $confidence,
            r.properties = $properties
        ON MATCH SET
            r.last_updated = datetime(),
            r.confidence = CASE
                WHEN $confidence > r.confidence THEN $confidence
                ELSE r.confidence
            END,
            r.properties = $properties
        SET
            r.data_sources = $data_sources,
            r.temporal_valid_from = $valid_from,
            r.temporal_valid_to = $valid_to
        RETURN r
        """

        parameters = {
            "relationship_id": relationship.relationship_id,
            "source_id": relationship.source_entity_id,
            "target_id": relationship.target_entity_id,
            "relationship_type": relationship.relationship_type.value,
            "confidence": relationship.confidence,
            "properties": relationship.properties,
            "data_sources": relationship.data_sources,
            "valid_from": relationship.temporal_valid_from.isoformat() if relationship.temporal_valid_from else None,
            "valid_to": relationship.temporal_valid_to.isoformat() if relationship.temporal_valid_to else None,
        }

        results = self.execute_query(query, parameters)
        return results[0] if results else {}

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """
        Retrieve entity by ID

        Args:
            entity_id: Entity identifier

        Returns:
            Entity object or None
        """
        query = """
        MATCH (e:Entity {entity_id: $entity_id})
        RETURN e
        """

        results = self.execute_query(query, {"entity_id": entity_id})

        if results:
            node_data = results[0]["e"]
            return self._node_to_entity(node_data)

        return None

    def find_entities_by_name(self, name: str, limit: int = 10) -> List[Entity]:
        """
        Find entities by name (fuzzy match)

        Args:
            name: Entity name to search
            limit: Maximum results to return

        Returns:
            List of matching entities
        """
        query = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $name OR e.original_name CONTAINS $name
        RETURN e
        ORDER BY e.confidence DESC
        LIMIT $limit
        """

        results = self.execute_query(query, {"name": name, "limit": limit})
        return [self._node_to_entity(r["e"]) for r in results]

    def get_entity_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None
    ) -> List[Tuple[Entity, Relationship]]:
        """
        Get neighboring entities and relationships

        Args:
            entity_id: Entity to get neighbors for
            depth: Depth of traversal
            relationship_types: Filter by relationship types

        Returns:
            List of (neighbor_entity, relationship) tuples
        """
        if relationship_types:
            rel_filter = f"[{','.join([':'+rt for rt in relationship_types])}]"
        else:
            rel_filter = ""

        query = f"""
        MATCH (e:Entity {{entity_id: $entity_id}})-[r{rel_filter}*1..{depth}]-(neighbor:Entity)
        RETURN DISTINCT neighbor, r
        LIMIT 100
        """

        results = self.execute_query(query, {"entity_id": entity_id})

        neighbors = []
        for result in results:
            neighbor_entity = self._node_to_entity(result["neighbor"])
            # Note: r is a list of relationships for paths > 1
            # For simplicity, we'll create a generic relationship
            neighbors.append((neighbor_entity, None))

        return neighbors

    def infer_transitive_ownership(self) -> int:
        """
        Infer transitive ownership relationships

        Returns:
            Number of relationships created
        """
        query = """
        MATCH (a:Entity)-[r1:RELATIONSHIP {relationship_type: 'OWNS'}]->(b:Entity)
             -[r2:RELATIONSHIP {relationship_type: 'OWNS'}]->(c:Entity)
        WHERE NOT (a)-[:RELATIONSHIP {relationship_type: 'INDIRECT_OWNS'}]->(c)
        WITH a, c,
             COALESCE(r1.properties.ownership_pct, 100.0) * COALESCE(r2.properties.ownership_pct, 100.0) / 100.0 AS indirect_pct
        WHERE indirect_pct >= 10.0
        CREATE (a)-[r:RELATIONSHIP {
            relationship_type: 'INDIRECT_OWNS',
            relationship_id: randomUUID(),
            properties: {ownership_pct: indirect_pct},
            derived: true,
            created_at: datetime()
        }]->(c)
        RETURN count(r) as count
        """

        results = self.execute_query(query)
        return results[0]["count"] if results else 0

    def find_shared_addresses(self) -> int:
        """
        Identify entities sharing addresses

        Returns:
            Number of relationships created
        """
        query = """
        MATCH (a:Entity), (b:Entity)
        WHERE a.entity_id < b.entity_id
        AND any(addr IN a.addresses WHERE addr IN b.addresses)
        AND NOT (a)-[:RELATIONSHIP {relationship_type: 'SHARES_ADDRESS'}]->(b)
        WITH a, b, [addr IN a.addresses WHERE addr IN b.addresses] as shared_addrs
        CREATE (a)-[r:RELATIONSHIP {
            relationship_type: 'SHARES_ADDRESS',
            relationship_id: randomUUID(),
            properties: {shared_addresses: shared_addrs},
            derived: true,
            created_at: datetime()
        }]->(b)
        RETURN count(r) as count
        """

        results = self.execute_query(query)
        return results[0]["count"] if results else 0

    def find_common_directors(self) -> int:
        """
        Identify companies sharing directors

        Returns:
            Number of relationships created
        """
        query = """
        MATCH (director:Entity {entity_type: 'INDIVIDUAL'})
        MATCH (director)-[r1:RELATIONSHIP {relationship_type: 'DIRECTOR_OF'}]->(company1:Entity)
        MATCH (director)-[r2:RELATIONSHIP {relationship_type: 'DIRECTOR_OF'}]->(company2:Entity)
        WHERE company1.entity_id < company2.entity_id
        AND NOT (company1)-[:RELATIONSHIP {relationship_type: 'SHARES_DIRECTOR'}]->(company2)
        CREATE (company1)-[r:RELATIONSHIP {
            relationship_type: 'SHARES_DIRECTOR',
            relationship_id: randomUUID(),
            properties: {common_director: director.name},
            derived: true,
            created_at: datetime()
        }]->(company2)
        RETURN count(r) as count
        """

        results = self.execute_query(query)
        return results[0]["count"] if results else 0

    def _node_to_entity(self, node_data: Dict[str, Any]) -> Entity:
        """Convert Neo4j node data to Entity object"""
        entity = Entity(
            entity_id=node_data.get("entity_id", ""),
            name=node_data.get("name", ""),
            original_name=node_data.get("original_name", ""),
            entity_type=EntityType(node_data.get("entity_type", "COMPANY")),
            identifiers=node_data.get("identifiers", {}),
            addresses=node_data.get("addresses", []),
            jurisdiction=node_data.get("jurisdiction"),
            phone_numbers=node_data.get("phone_numbers", []),
            email_addresses=node_data.get("email_addresses", []),
            websites=node_data.get("websites", []),
            industry_codes=node_data.get("industry_codes", []),
            roles=node_data.get("roles", []),
            confidence=node_data.get("confidence", 1.0),
            data_sources=node_data.get("data_sources", []),
        )

        return entity

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
