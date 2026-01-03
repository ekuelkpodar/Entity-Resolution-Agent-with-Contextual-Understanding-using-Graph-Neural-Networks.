"""
FastAPI application for Entity Resolution Agent
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.entity_resolution.core.entities import Entity, EntityType
from src.entity_resolution.core.entity_extractor import EntityExtractor
from src.entity_resolution.core.entity_normalizer import EntityNormalizer
from src.entity_resolution.core.decision_engine import EntityResolutionDecisionEngine
from src.entity_resolution.core.beneficial_ownership import BeneficialOwnershipResolver
from src.entity_resolution.core.graph_db import Neo4jConnection
from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="Entity Resolution Agent API",
    description="Production-grade entity resolution using GNNs and contextual understanding",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class EntityCreate(BaseModel):
    """Entity creation request"""
    name: str
    entity_type: str = Field(default="COMPANY")
    jurisdiction: Optional[str] = None
    addresses: List[str] = Field(default_factory=list)
    identifiers: dict = Field(default_factory=dict)


class EntityResponse(BaseModel):
    """Entity response"""
    entity_id: str
    name: str
    entity_type: str
    confidence: float


class EntityResolutionRequest(BaseModel):
    """Entity resolution request"""
    entity1: EntityCreate
    entity2: EntityCreate


class EntityResolutionResponse(BaseModel):
    """Entity resolution response"""
    decision: str
    confidence: float
    action: str
    explanation: str


class BeneficialOwnershipRequest(BaseModel):
    """Beneficial ownership request"""
    entity_id: str
    max_depth: int = Field(default=10, ge=1, le=20)


class BeneficialOwnershipResponse(BaseModel):
    """Beneficial ownership response"""
    entity_id: str
    ubos: List[dict]
    complexity_score: float
    risk_level: str
    transparency: str


class DocumentExtractionRequest(BaseModel):
    """Document extraction request"""
    text: str
    source: str = "api"


class DocumentExtractionResponse(BaseModel):
    """Document extraction response"""
    entities: List[dict]
    num_entities: int


# Dependency injection for services
def get_graph_db():
    """Get Neo4j database connection"""
    try:
        db = Neo4jConnection()
        yield db
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    finally:
        if 'db' in locals():
            db.close()


def get_decision_engine():
    """Get entity resolution decision engine"""
    return EntityResolutionDecisionEngine()


def get_normalizer():
    """Get entity normalizer"""
    return EntityNormalizer()


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Entity Resolution Agent API",
        "version": "0.1.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0"
    }


@app.post("/api/v1/resolve", response_model=EntityResolutionResponse)
async def resolve_entities(
    request: EntityResolutionRequest,
    engine: EntityResolutionDecisionEngine = Depends(get_decision_engine)
):
    """
    Resolve whether two entities are the same

    Args:
        request: Entity resolution request with two entities

    Returns:
        Entity resolution decision
    """
    try:
        # Convert request entities to Entity objects
        entity1 = Entity(
            name=request.entity1.name,
            entity_type=EntityType[request.entity1.entity_type.upper()],
            jurisdiction=request.entity1.jurisdiction,
            addresses=request.entity1.addresses,
            identifiers=request.entity1.identifiers
        )

        entity2 = Entity(
            name=request.entity2.name,
            entity_type=EntityType[request.entity2.entity_type.upper()],
            jurisdiction=request.entity2.jurisdiction,
            addresses=request.entity2.addresses,
            identifiers=request.entity2.identifiers
        )

        # Resolve entities
        decision = await engine.resolve_entities(entity1, entity2)

        return EntityResolutionResponse(
            decision=decision.decision,
            confidence=decision.confidence,
            action=decision.action,
            explanation=decision.explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/beneficial-ownership", response_model=BeneficialOwnershipResponse)
async def find_beneficial_owners(
    request: BeneficialOwnershipRequest,
    graph_db: Neo4jConnection = Depends(get_graph_db)
):
    """
    Find ultimate beneficial owners of an entity

    Args:
        request: Beneficial ownership request

    Returns:
        UBO information
    """
    try:
        resolver = BeneficialOwnershipResolver(graph_db)

        # Analyze ownership structure
        analysis = await resolver.analyze_ownership_structure(request.entity_id)

        return BeneficialOwnershipResponse(
            entity_id=analysis["entity_id"],
            ubos=analysis["ubos"],
            complexity_score=analysis["complexity_score"],
            risk_level=analysis["risk_level"],
            transparency=analysis["transparency"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/extract-entities", response_model=DocumentExtractionResponse)
async def extract_entities(request: DocumentExtractionRequest):
    """
    Extract entities from document text

    Args:
        request: Document extraction request

    Returns:
        Extracted entities
    """
    try:
        from src.entity_resolution.core.entities import Document

        # Create extractor (without LLM for now)
        extractor = EntityExtractor(llm_client=None)

        # Create document
        document = Document(
            text=request.text,
            source=request.source
        )

        # Extract entities
        entities = await extractor.extract_entities(document)

        # Convert to dictionaries
        entities_data = [
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "entity_type": e.entity_type.value,
                "confidence": e.confidence,
                "addresses": e.addresses,
                "identifiers": e.identifiers
            }
            for e in entities
        ]

        return DocumentExtractionResponse(
            entities=entities_data,
            num_entities=len(entities_data)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/entities", response_model=EntityResponse)
async def create_entity(
    entity: EntityCreate,
    graph_db: Neo4jConnection = Depends(get_graph_db),
    normalizer: EntityNormalizer = Depends(get_normalizer)
):
    """
    Create a new entity in the knowledge graph

    Args:
        entity: Entity to create
        graph_db: Database connection
        normalizer: Entity normalizer

    Returns:
        Created entity
    """
    try:
        # Create Entity object
        new_entity = Entity(
            name=entity.name,
            entity_type=EntityType[entity.entity_type.upper()],
            jurisdiction=entity.jurisdiction,
            addresses=entity.addresses,
            identifiers=entity.identifiers
        )

        # Normalize entity
        new_entity = normalizer.normalize(new_entity)

        # Save to database
        graph_db.create_entity_node(new_entity)

        return EntityResponse(
            entity_id=new_entity.entity_id,
            name=new_entity.name,
            entity_type=new_entity.entity_type.value,
            confidence=new_entity.confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    graph_db: Neo4jConnection = Depends(get_graph_db)
):
    """
    Retrieve entity by ID

    Args:
        entity_id: Entity identifier
        graph_db: Database connection

    Returns:
        Entity data
    """
    try:
        entity = graph_db.get_entity_by_id(entity_id)

        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        return EntityResponse(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            confidence=entity.confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/entities/search/{name}")
async def search_entities(
    name: str,
    limit: int = 10,
    graph_db: Neo4jConnection = Depends(get_graph_db)
):
    """
    Search for entities by name

    Args:
        name: Name to search for
        limit: Maximum number of results
        graph_db: Database connection

    Returns:
        List of matching entities
    """
    try:
        entities = graph_db.find_entities_by_name(name, limit)

        return {
            "query": name,
            "count": len(entities),
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "entity_type": e.entity_type.value,
                    "confidence": e.confidence
                }
                for e in entities
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/graph/infer-relationships")
async def infer_relationships(graph_db: Neo4jConnection = Depends(get_graph_db)):
    """
    Infer derived relationships in the graph

    Args:
        graph_db: Database connection

    Returns:
        Number of relationships inferred
    """
    try:
        # Infer transitive ownership
        ownership_count = graph_db.infer_transitive_ownership()

        # Find shared addresses
        address_count = graph_db.find_shared_addresses()

        # Find common directors
        director_count = graph_db.find_common_directors()

        return {
            "transitive_ownership": ownership_count,
            "shared_addresses": address_count,
            "common_directors": director_count,
            "total": ownership_count + address_count + director_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=1 if settings.api_reload else settings.api_workers
    )
