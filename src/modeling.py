from __future__ import annotations

import json
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.sample_data import ensure_courtlistener_sample_dataset


def _build_graph(base_dir: Path) -> tuple[nx.Graph, pd.DataFrame]:
    dataset = ensure_courtlistener_sample_dataset(base_dir)
    process_df = pd.read_csv(dataset["dockets_path"])
    parties_df = pd.read_csv(dataset["parties_path"])
    lawyers_df = pd.read_csv(dataset["attorneys_path"])
    judges_df = pd.read_csv(dataset["judges_path"])
    edges_df = pd.read_csv(dataset["edges_path"])

    graph = nx.Graph()

    for _, row in process_df.iterrows():
        graph.add_node(row["docket_id"], node_type="docket", **row.to_dict())
    for _, row in parties_df.iterrows():
        graph.add_node(row["party_id"], node_type="party", **row.to_dict())
    for _, row in lawyers_df.iterrows():
        graph.add_node(row["attorney_id"], node_type="attorney", **row.to_dict())
    for _, row in judges_df.iterrows():
        graph.add_node(row["judge_id"], node_type="judge", **row.to_dict())
    for _, row in edges_df.iterrows():
        graph.add_edge(row["source_id"], row["target_id"], edge_type=row["edge_type"])

    return graph, process_df


def _extract_process_features(graph: nx.Graph, process_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in process_df.iterrows():
        process_id = row["docket_id"]
        neighbors = list(graph.neighbors(process_id))
        party_neighbors = [n for n in neighbors if graph.nodes[n].get("node_type") == "party"]
        attorney_neighbors = []
        judge_neighbors = [n for n in neighbors if graph.nodes[n].get("node_type") == "judge"]
        for party_id in party_neighbors:
            attorney_neighbors.extend(
                neigh for neigh in graph.neighbors(party_id) if graph.nodes[neigh].get("node_type") == "attorney"
            )
        recurring_process_links = 0
        for party_id in party_neighbors:
            recurring_process_links += sum(
                1
                for neigh in graph.neighbors(party_id)
                if neigh != process_id and graph.nodes[neigh].get("node_type") == "docket"
            )

        rows.append(
            {
                "docket_id": process_id,
                "claim_value": float(row["claim_value"]),
                "repeat_player_signal": int(row["repeat_player_signal"]),
                "negative_precedent_signal": int(row["negative_precedent_signal"]),
                "party_degree": int(len(party_neighbors)),
                "attorney_degree": int(len(set(attorney_neighbors))),
                "judge_degree": int(len(set(judge_neighbors))),
                "related_docket_links": int(recurring_process_links),
                "nature_credit": int(row["nature_of_suit"] == "consumer_credit"),
                "nature_goods": int(row["nature_of_suit"] == "consumer_goods"),
                "nature_healthcare": int(row["nature_of_suit"] == "healthcare"),
                "court_sdny": int(row["court"] == "S.D.N.Y."),
                "court_ndcal": int(row["court"] == "N.D. Cal."),
                "court_dmass": int(row["court"] == "D. Mass."),
                "settled": int(row["settled"]),
            }
        )
    return pd.DataFrame(rows)


def _recommend_settlement_band(probability: float, claim_value: float) -> str:
    if probability >= 0.65:
        return f"strong_settlement_signal | suggested_band={round(claim_value * 0.55, 2)}"
    if probability >= 0.45:
        return f"moderate_settlement_signal | suggested_band={round(claim_value * 0.42, 2)}"
    return "weak_settlement_signal | escalate_to_legal_review"


def run_pipeline(base_dir: str | Path) -> dict:
    base_path = Path(base_dir)
    graph, process_df = _build_graph(base_path)
    features_df = _extract_process_features(graph, process_df)

    X = features_df.drop(columns=["docket_id", "settled"])
    y = features_df["settled"]

    X_train, X_test, y_train, y_test, _train_ids, test_ids = train_test_split(
        X, y, features_df["docket_id"], test_size=0.33, random_state=42, stratify=y
    )

    runtime_mode = "graph_feature_fallback"
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401

        runtime_mode = "gnn_ready_graph_feature_benchmark"
    except Exception:
        runtime_mode = "graph_feature_fallback"

    model = RandomForestClassifier(n_estimators=180, random_state=42)
    model.fit(X_train, y_train)
    predicted_labels = model.predict(X_test)
    predicted_probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predicted_labels)
    macro_f1 = f1_score(y_test, predicted_labels, average="macro")
    roc_auc = roc_auc_score(y_test, predicted_probabilities)

    decisions = []
    for process_id, probability in zip(test_ids, predicted_probabilities, strict=False):
        claim_value = float(features_df.loc[features_df["docket_id"] == process_id, "claim_value"].iloc[0])
        decisions.append(
            {
                "docket_id": process_id,
                "settlement_probability": round(float(probability), 4),
                "recommendation": _recommend_settlement_band(float(probability), claim_value),
            }
        )

    artifacts_dir = base_path / "artifacts"
    processed_dir = base_path / "data" / "processed"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_artifact = processed_dir / "process_feature_table.csv"
    decision_artifact = processed_dir / "settlement_support.csv"
    report_artifact = processed_dir / "process_linkage_settlement_report.json"
    model_artifact = artifacts_dir / "graph_settlement_model.joblib"

    features_df.to_csv(feature_artifact, index=False)
    pd.DataFrame(decisions).to_csv(decision_artifact, index=False)
    joblib.dump(model, model_artifact)

    summary = {
        "runtime_mode": runtime_mode,
        "dataset_source": "courtlistener_style_sample",
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "docket_count": int(len(process_df)),
        "linked_docket_groups": int(features_df["related_docket_links"].gt(0).sum()),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "feature_artifact": str(feature_artifact),
        "decision_artifact": str(decision_artifact),
        "model_artifact": str(model_artifact),
        "report_artifact": str(report_artifact),
    }
    report_artifact.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
