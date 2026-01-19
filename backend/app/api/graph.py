from fastapi import APIRouter, HTTPException
from typing import List

from app.models.graph import (
    Anchor,
    AnchorCreate,
    Edge,
    EdgeCreate,
    Evidence,
    EvidenceCreate,
    GraphView,
    Node,
    NodeCreate,
)
from app.services.graph_store import graph_store

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("", response_model=GraphView)
def get_graph() -> GraphView:
    return graph_store.get_graph()


@router.post("/nodes", response_model=Node, status_code=201)
def create_node(payload: NodeCreate) -> Node:
    return graph_store.add_node(payload)


@router.get("/nodes", response_model=List[Node])
def list_nodes() -> List[Node]:
    return graph_store.list_nodes()


@router.get("/nodes/{node_id}", response_model=Node)
def get_node(node_id: str) -> Node:
    try:
        return graph_store.get_node(node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/edges", response_model=Edge, status_code=201)
def create_edge(payload: EdgeCreate) -> Edge:
    try:
        return graph_store.add_edge(payload)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/edges", response_model=List[Edge])
def list_edges() -> List[Edge]:
    return graph_store.list_edges()


@router.get("/edges/{edge_id}", response_model=Edge)
def get_edge(edge_id: str) -> Edge:
    try:
        return graph_store.get_edge(edge_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/anchors", response_model=Anchor, status_code=201)
def create_anchor(payload: AnchorCreate) -> Anchor:
    try:
        return graph_store.add_anchor(payload)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/anchors", response_model=List[Anchor])
def list_anchors() -> List[Anchor]:
    return graph_store.list_anchors()


@router.get("/anchors/{anchor_id}", response_model=Anchor)
def get_anchor(anchor_id: str) -> Anchor:
    try:
        return graph_store.get_anchor(anchor_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/evidence", response_model=Evidence, status_code=201)
def create_evidence(payload: EvidenceCreate) -> Evidence:
    try:
        return graph_store.add_evidence(payload)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/evidence", response_model=List[Evidence])
def list_evidence() -> List[Evidence]:
    return graph_store.list_evidence()


@router.get("/evidence/{evidence_id}", response_model=Evidence)
def get_evidence(evidence_id: str) -> Evidence:
    try:
        return graph_store.get_evidence(evidence_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
