from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List
from uuid import uuid4

from app.models.graph import (
    Anchor,
    AnchorCreate,
    Edge,
    EdgeCreate,
    Evidence,
    EvidenceCreate,
    EvidenceTarget,
    GraphView,
    Node,
    NodeCreate,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GraphStore:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.anchors: Dict[str, Anchor] = {}
        self.evidence: Dict[str, Evidence] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, payload: NodeCreate) -> Node:
        node_id = str(uuid4())
        now = _utc_now()
        node = Node(
            id=node_id,
            created_at=now,
            updated_at=now,
            **payload.dict(),
        )
        self.nodes[node_id] = node
        self.adjacency.setdefault(node_id, [])
        return node

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def list_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def add_edge(self, payload: EdgeCreate) -> Edge:
        if payload.from_node_id not in self.nodes or payload.to_node_id not in self.nodes:
            missing = [
                node_id
                for node_id in (payload.from_node_id, payload.to_node_id)
                if node_id not in self.nodes
            ]
            raise KeyError(f"Missing nodes: {', '.join(missing)}")
        if payload.from_node_id == payload.to_node_id:
            raise ValueError("Edge endpoints must be different nodes.")

        edge_id = str(uuid4())
        now = _utc_now()
        edge = Edge(
            id=edge_id,
            created_at=now,
            updated_at=now,
            **payload.dict(),
        )
        self.edges[edge_id] = edge
        self._link_edge(edge)
        return edge

    def _link_edge(self, edge: Edge) -> None:
        for node_id in (edge.from_node_id, edge.to_node_id):
            edge_list = self.adjacency.setdefault(node_id, [])
            if edge.id not in edge_list:
                edge_list.append(edge.id)

    def get_edge(self, edge_id: str) -> Edge:
        return self.edges[edge_id]

    def list_edges(self) -> List[Edge]:
        return list(self.edges.values())

    def add_anchor(self, payload: AnchorCreate) -> Anchor:
        if payload.node_id not in self.nodes:
            raise KeyError(f"Missing node: {payload.node_id}")
        anchor_id = str(uuid4())
        anchor = Anchor(
            id=anchor_id,
            created_at=_utc_now(),
            **payload.dict(),
        )
        self.anchors[anchor_id] = anchor
        return anchor

    def get_anchor(self, anchor_id: str) -> Anchor:
        return self.anchors[anchor_id]

    def list_anchors(self) -> List[Anchor]:
        return list(self.anchors.values())

    def add_evidence(self, payload: EvidenceCreate) -> Evidence:
        if payload.target_type == EvidenceTarget.node and payload.target_id not in self.nodes:
            raise KeyError(f"Missing node: {payload.target_id}")
        if payload.target_type == EvidenceTarget.edge and payload.target_id not in self.edges:
            raise KeyError(f"Missing edge: {payload.target_id}")

        evidence_id = str(uuid4())
        evidence = Evidence(
            id=evidence_id,
            created_at=_utc_now(),
            **payload.dict(),
        )
        self.evidence[evidence_id] = evidence
        return evidence

    def get_evidence(self, evidence_id: str) -> Evidence:
        return self.evidence[evidence_id]

    def list_evidence(self) -> List[Evidence]:
        return list(self.evidence.values())

    def get_graph(self) -> GraphView:
        return GraphView(
            nodes=self.nodes,
            edges=self.edges,
            adjacency=dict(self.adjacency),
        )


graph_store = GraphStore()
