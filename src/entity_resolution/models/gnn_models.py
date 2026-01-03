"""
Graph Neural Network models for entity resolution
"""
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Data = None
    Batch = None

import numpy as np
from config.settings import settings


class EntityResolutionGNN(nn.Module):
    """
    Graph Neural Network for entity resolution
    Uses graph structure to generate entity embeddings
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        use_gat: bool = True
    ):
        """
        Initialize GNN model

        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            num_layers: Number of graph convolution layers
            dropout: Dropout rate
            use_gat: Use Graph Attention Networks instead of GraphSAGE
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and PyTorch Geometric are required. Install with: pip install torch torch-geometric")

        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim or settings.gnn_hidden_dim
        self.embedding_dim = embedding_dim or settings.gnn_embedding_dim
        self.num_layers = num_layers or settings.gnn_num_layers
        self.dropout = dropout or settings.gnn_dropout
        self.use_gat = use_gat

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()

        for i in range(self.num_layers):
            if use_gat:
                # Graph Attention Network layers
                if i == 0:
                    self.conv_layers.append(
                        GATConv(
                            self.hidden_dim,
                            self.hidden_dim,
                            heads=4,
                            dropout=self.dropout
                        )
                    )
                else:
                    self.conv_layers.append(
                        GATConv(
                            self.hidden_dim * 4,  # 4 attention heads
                            self.hidden_dim,
                            heads=4,
                            dropout=self.dropout
                        )
                    )
            else:
                # GraphSAGE layers
                if i == 0:
                    self.conv_layers.append(
                        SAGEConv(self.hidden_dim, self.hidden_dim)
                    )
                else:
                    self.conv_layers.append(
                        SAGEConv(self.hidden_dim, self.hidden_dim)
                    )

        # Final embedding layer
        final_input_dim = self.hidden_dim * 4 if use_gat else self.hidden_dim

        self.embedding_layer = nn.Sequential(
            nn.Linear(final_input_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GNN

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feature_dim] (optional)

        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        x, edge_index = data.x, data.edge_index

        # Encode node features
        x = self.node_encoder(x)

        # Apply graph convolutions
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:  # Don't apply ReLU after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Generate final embeddings
        embeddings = self.embedding_layer(x)
        embeddings = self.layer_norm(embeddings)

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class EntityMatchingModel(nn.Module):
    """
    Siamese network for entity matching using GNN embeddings
    """

    def __init__(self, embedding_dim: Optional[int] = None):
        """
        Initialize entity matching model

        Args:
            embedding_dim: Dimension of input embeddings
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        super().__init__()

        self.embedding_dim = embedding_dim or settings.gnn_embedding_dim

        # Matching network
        self.matcher = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute match probability between two entity embeddings

        Args:
            emb1: Entity 1 embeddings [batch_size, embedding_dim]
            emb2: Entity 2 embeddings [batch_size, embedding_dim]

        Returns:
            Match probabilities [batch_size, 1]
        """
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)

        # Compute match score
        match_score = self.matcher(combined)

        return match_score

    def predict(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict entity matches

        Args:
            emb1: Entity 1 embeddings
            emb2: Entity 2 embeddings
            threshold: Classification threshold

        Returns:
            Tuple of (match_probabilities, binary_predictions)
        """
        match_probs = self.forward(emb1, emb2)
        binary_preds = (match_probs > threshold).float()

        return match_probs, binary_preds


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for entity matching
    Brings matching entities closer, pushes non-matching entities apart
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss

        Args:
            margin: Margin for non-matching pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            emb1: First entity embeddings [batch_size, embedding_dim]
            emb2: Second entity embeddings [batch_size, embedding_dim]
            labels: Match labels (1 for match, 0 for non-match) [batch_size]

        Returns:
            Loss value
        """
        # Compute euclidean distance
        distances = F.pairwise_distance(emb1, emb2)

        # Contrastive loss
        # For matching pairs: minimize distance
        # For non-matching pairs: maximize distance (up to margin)
        loss_match = labels * torch.pow(distances, 2)
        loss_non_match = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        loss = torch.mean(loss_match + loss_non_match)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for entity matching
    anchor-positive-negative triplets
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss

        Args:
            margin: Margin between positive and negative pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss

        Args:
            anchor: Anchor entity embeddings [batch_size, embedding_dim]
            positive: Positive (matching) entity embeddings [batch_size, embedding_dim]
            negative: Negative (non-matching) entity embeddings [batch_size, embedding_dim]

        Returns:
            Loss value
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)

        # Triplet loss: d(anchor, positive) + margin < d(anchor, negative)
        loss = torch.mean(torch.clamp(pos_distance - neg_distance + self.margin, min=0.0))

        return loss


class GraphFeatureEncoder:
    """
    Encode entity and relationship features for GNN
    """

    def __init__(self):
        """Initialize feature encoder"""
        self.node_feature_dim = 164  # Total feature dimension

    def encode_node_features(self, entity: "Entity") -> np.ndarray:
        """
        Create feature vector for entity node

        Args:
            entity: Entity object

        Returns:
            Feature vector
        """
        from src.entity_resolution.core.entities import EntityType
        from datetime import datetime

        features = []

        # 1. Entity type one-hot (4 dims)
        entity_type_map = {
            EntityType.COMPANY: 0,
            EntityType.INDIVIDUAL: 1,
            EntityType.GOVERNMENT: 2,
            EntityType.NGO: 3
        }
        entity_type_vec = np.zeros(4)
        entity_type_vec[entity_type_map[entity.entity_type]] = 1
        features.append(entity_type_vec)

        # 2. Has identifiers flags (5 dims)
        identifier_flags = [
            float("ein" in entity.identifiers),
            float("lei" in entity.identifiers),
            float("company_number" in entity.identifiers),
            float("ssn" in entity.identifiers),
            float(len(entity.identifiers) > 0)
        ]
        features.append(identifier_flags)

        # 3. Entity age (1 dim)
        if entity.incorporation_date:
            age_years = (datetime.now() - entity.incorporation_date).days / 365.25
            age_normalized = min(age_years / 100, 1.0)
        else:
            age_normalized = 0.0
        features.append([age_normalized])

        # 4. Activity indicators (3 dims)
        is_active = float(entity.dissolution_date is None)
        has_contact = float(len(entity.phone_numbers) > 0 or len(entity.email_addresses) > 0)
        has_website = float(len(entity.websites) > 0)
        features.append([is_active, has_contact, has_website])

        # 5. Address count (normalized) (1 dim)
        addr_count = min(len(entity.addresses) / 5.0, 1.0)
        features.append([addr_count])

        # 6. Confidence score (1 dim)
        features.append([entity.confidence])

        # 7. Name length features (2 dims)
        name_len_normalized = min(len(entity.name) / 100.0, 1.0)
        name_word_count = len(entity.name.split())
        name_word_count_normalized = min(name_word_count / 10.0, 1.0)
        features.append([name_len_normalized, name_word_count_normalized])

        # 8. Role features (1 dim)
        has_roles = float(len(entity.roles) > 0)
        features.append([has_roles])

        # 9. Padding to reach desired dimension (147 dims)
        # This can be replaced with actual text embeddings in production
        padding = np.zeros(147)
        features.append(padding)

        # Concatenate all features
        feature_vector = np.concatenate(features)

        assert len(feature_vector) == self.node_feature_dim, \
            f"Feature vector dimension mismatch: {len(feature_vector)} != {self.node_feature_dim}"

        return feature_vector.astype(np.float32)

    def encode_edge_features(self, relationship: "Relationship") -> np.ndarray:
        """
        Create feature vector for relationship edge

        Args:
            relationship: Relationship object

        Returns:
            Feature vector
        """
        from src.entity_resolution.core.entities import RelationshipType
        from datetime import datetime

        features = []

        # 1. Relationship type one-hot (12 dims)
        rel_types = [rt.value for rt in RelationshipType]
        rel_type_vec = np.zeros(len(rel_types))
        try:
            idx = rel_types.index(relationship.relationship_type.value)
            rel_type_vec[idx] = 1
        except ValueError:
            pass
        features.append(rel_type_vec)

        # 2. Ownership percentage (1 dim)
        ownership_pct = relationship.properties.get("ownership_pct", 0.0)
        features.append([ownership_pct / 100.0])

        # 3. Confidence score (1 dim)
        features.append([relationship.confidence])

        # 4. Temporal features (2 dims)
        if relationship.temporal_valid_from:
            days_active = (datetime.now() - relationship.temporal_valid_from).days
            temporal_normalized = min(days_active / 3650.0, 1.0)
        else:
            temporal_normalized = 0.0
        features.append([temporal_normalized])

        is_current = float(relationship.temporal_valid_to is None)
        features.append([is_current])

        # Concatenate all features
        feature_vector = np.concatenate(features)

        return feature_vector.astype(np.float32)
