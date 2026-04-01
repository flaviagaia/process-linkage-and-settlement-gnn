from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


PROCESSES = [
    ("PROC-1001", "bancario", "conhecimento", 4200, 1, 1, 0),
    ("PROC-1002", "bancario", "conhecimento", 4500, 1, 1, 1),
    ("PROC-1003", "consumidor", "instrucao", 1800, 0, 0, 1),
    ("PROC-1004", "consumidor", "instrucao", 2100, 0, 0, 1),
    ("PROC-1005", "saude", "recursal", 12000, 1, 0, 0),
    ("PROC-1006", "saude", "conhecimento", 9800, 1, 0, 0),
    ("PROC-1007", "bancario", "instrucao", 3900, 1, 1, 1),
    ("PROC-1008", "consumidor", "conhecimento", 1600, 0, 0, 1),
    ("PROC-1009", "saude", "recursal", 14000, 1, 1, 0),
    ("PROC-1010", "bancario", "recursal", 7000, 1, 1, 0),
]

PARTIES = [
    ("PARTE-1001", "autor", "joao_silva"),
    ("PARTE-1002", "autor", "maria_souza"),
    ("PARTE-1003", "reu", "banco_alfa"),
    ("PARTE-1004", "reu", "varejo_beta"),
    ("PARTE-1005", "reu", "seguradora_gama"),
    ("PARTE-1006", "autor", "lucas_ferreira"),
]

LAWYERS = [
    ("ADV-1001", "escritorio_justica"),
    ("ADV-1002", "advocacia_cidada"),
    ("ADV-1003", "contencioso_alfa"),
]

EDGES = [
    ("PARTE-1001", "PROC-1001", "parte_em"),
    ("PARTE-1001", "PROC-1002", "parte_em"),
    ("PARTE-1002", "PROC-1003", "parte_em"),
    ("PARTE-1002", "PROC-1004", "parte_em"),
    ("PARTE-1002", "PROC-1006", "parte_em"),
    ("PARTE-1006", "PROC-1007", "parte_em"),
    ("PARTE-1006", "PROC-1008", "parte_em"),
    ("PARTE-1001", "PROC-1010", "parte_em"),
    ("PARTE-1003", "PROC-1001", "parte_em"),
    ("PARTE-1003", "PROC-1002", "parte_em"),
    ("PARTE-1003", "PROC-1007", "parte_em"),
    ("PARTE-1003", "PROC-1010", "parte_em"),
    ("PARTE-1004", "PROC-1003", "parte_em"),
    ("PARTE-1004", "PROC-1004", "parte_em"),
    ("PARTE-1004", "PROC-1008", "parte_em"),
    ("PARTE-1005", "PROC-1005", "parte_em"),
    ("PARTE-1005", "PROC-1006", "parte_em"),
    ("PARTE-1005", "PROC-1009", "parte_em"),
    ("ADV-1001", "PARTE-1001", "representa"),
    ("ADV-1001", "PARTE-1002", "representa"),
    ("ADV-1001", "PARTE-1006", "representa"),
    ("ADV-1002", "PARTE-1003", "representa"),
    ("ADV-1002", "PARTE-1004", "representa"),
    ("ADV-1003", "PARTE-1005", "representa"),
]


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".csv", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        df.to_csv(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def ensure_graph_dataset(base_dir: str | Path) -> dict[str, str]:
    base_path = Path(base_dir)
    raw_dir = base_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    process_path = raw_dir / "processes.csv"
    parties_path = raw_dir / "parties.csv"
    lawyers_path = raw_dir / "lawyers.csv"
    edges_path = raw_dir / "edges.csv"

    process_df = pd.DataFrame(
        PROCESSES,
        columns=[
            "process_id",
            "theme",
            "phase",
            "claim_value",
            "recurring_party",
            "negative_precedent",
            "settled",
        ],
    )
    parties_df = pd.DataFrame(PARTIES, columns=["party_id", "party_role", "party_name"])
    lawyers_df = pd.DataFrame(LAWYERS, columns=["lawyer_id", "office_name"])
    edges_df = pd.DataFrame(EDGES, columns=["source_id", "target_id", "edge_type"])

    _atomic_write(process_df, process_path)
    _atomic_write(parties_df, parties_path)
    _atomic_write(lawyers_df, lawyers_path)
    _atomic_write(edges_df, edges_path)

    return {
        "process_path": str(process_path),
        "parties_path": str(parties_path),
        "lawyers_path": str(lawyers_path),
        "edges_path": str(edges_path),
    }
