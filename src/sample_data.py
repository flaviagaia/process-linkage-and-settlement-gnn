from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


DOCKETS = [
    ("DKT-1001", "consumer_credit", "S.D.N.Y.", "banco_alfa_v_joao", 4200, 1, 1, 0),
    ("DKT-1002", "consumer_credit", "S.D.N.Y.", "banco_alfa_v_joao_2", 4500, 1, 1, 1),
    ("DKT-1003", "consumer_goods", "N.D. Cal.", "varejo_beta_v_maria", 1800, 0, 0, 1),
    ("DKT-1004", "consumer_goods", "N.D. Cal.", "varejo_beta_v_maria_2", 2100, 0, 0, 1),
    ("DKT-1005", "healthcare", "D. Mass.", "seguradora_gama_v_carla", 12000, 1, 0, 0),
    ("DKT-1006", "healthcare", "D. Mass.", "seguradora_gama_v_maria", 9800, 1, 0, 0),
    ("DKT-1007", "consumer_credit", "S.D.N.Y.", "banco_alfa_v_lucas", 3900, 1, 1, 1),
    ("DKT-1008", "consumer_goods", "N.D. Cal.", "varejo_beta_v_lucas", 1600, 0, 0, 1),
    ("DKT-1009", "healthcare", "D. Mass.", "seguradora_gama_v_carla_2", 14000, 1, 1, 0),
    ("DKT-1010", "consumer_credit", "S.D.N.Y.", "banco_alfa_v_joao_appeal", 7000, 1, 1, 0),
]

PARTIES = [
    ("PTY-1001", "Plaintiff", "joao_silva"),
    ("PTY-1002", "Plaintiff", "maria_souza"),
    ("PTY-1003", "Defendant", "banco_alfa"),
    ("PTY-1004", "Defendant", "varejo_beta"),
    ("PTY-1005", "Defendant", "seguradora_gama"),
    ("PTY-1006", "Plaintiff", "lucas_ferreira"),
    ("PTY-1007", "Plaintiff", "carla_mendes"),
]

ATTORNEYS = [
    ("ATTY-1001", "escritorio_justica"),
    ("ATTY-1002", "advocacia_cidada"),
    ("ATTY-1003", "contencioso_alfa"),
]

JUDGES = [
    ("JDG-1001", "judge_hamilton"),
    ("JDG-1002", "judge_reyes"),
    ("JDG-1003", "judge_bennett"),
]

EDGES = [
    ("PTY-1001", "DKT-1001", "party_in_docket"),
    ("PTY-1001", "DKT-1002", "party_in_docket"),
    ("PTY-1001", "DKT-1010", "party_in_docket"),
    ("PTY-1002", "DKT-1003", "party_in_docket"),
    ("PTY-1002", "DKT-1004", "party_in_docket"),
    ("PTY-1002", "DKT-1006", "party_in_docket"),
    ("PTY-1003", "DKT-1001", "party_in_docket"),
    ("PTY-1003", "DKT-1002", "party_in_docket"),
    ("PTY-1003", "DKT-1007", "party_in_docket"),
    ("PTY-1003", "DKT-1010", "party_in_docket"),
    ("PTY-1004", "DKT-1003", "party_in_docket"),
    ("PTY-1004", "DKT-1004", "party_in_docket"),
    ("PTY-1004", "DKT-1008", "party_in_docket"),
    ("PTY-1005", "DKT-1005", "party_in_docket"),
    ("PTY-1005", "DKT-1006", "party_in_docket"),
    ("PTY-1005", "DKT-1009", "party_in_docket"),
    ("PTY-1006", "DKT-1007", "party_in_docket"),
    ("PTY-1006", "DKT-1008", "party_in_docket"),
    ("PTY-1007", "DKT-1005", "party_in_docket"),
    ("PTY-1007", "DKT-1009", "party_in_docket"),
    ("ATTY-1001", "PTY-1001", "represents"),
    ("ATTY-1001", "PTY-1002", "represents"),
    ("ATTY-1001", "PTY-1006", "represents"),
    ("ATTY-1001", "PTY-1007", "represents"),
    ("ATTY-1002", "PTY-1003", "represents"),
    ("ATTY-1002", "PTY-1004", "represents"),
    ("ATTY-1003", "PTY-1005", "represents"),
    ("JDG-1001", "DKT-1001", "assigned_to"),
    ("JDG-1001", "DKT-1002", "assigned_to"),
    ("JDG-1001", "DKT-1007", "assigned_to"),
    ("JDG-1001", "DKT-1010", "assigned_to"),
    ("JDG-1002", "DKT-1003", "assigned_to"),
    ("JDG-1002", "DKT-1004", "assigned_to"),
    ("JDG-1002", "DKT-1008", "assigned_to"),
    ("JDG-1003", "DKT-1005", "assigned_to"),
    ("JDG-1003", "DKT-1006", "assigned_to"),
    ("JDG-1003", "DKT-1009", "assigned_to"),
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


def ensure_courtlistener_sample_dataset(base_dir: str | Path) -> dict[str, str]:
    base_path = Path(base_dir)
    raw_dir = base_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dockets_path = raw_dir / "dockets.csv"
    parties_path = raw_dir / "parties.csv"
    attorneys_path = raw_dir / "attorneys.csv"
    judges_path = raw_dir / "judges.csv"
    edges_path = raw_dir / "edges.csv"

    dockets_df = pd.DataFrame(
        DOCKETS,
        columns=[
            "docket_id",
            "nature_of_suit",
            "court",
            "slug",
            "claim_value",
            "repeat_player_signal",
            "negative_precedent_signal",
            "settled",
        ],
    )
    parties_df = pd.DataFrame(PARTIES, columns=["party_id", "party_role", "party_name"])
    attorneys_df = pd.DataFrame(ATTORNEYS, columns=["attorney_id", "office_name"])
    judges_df = pd.DataFrame(JUDGES, columns=["judge_id", "judge_name"])
    edges_df = pd.DataFrame(EDGES, columns=["source_id", "target_id", "edge_type"])

    _atomic_write(dockets_df, dockets_path)
    _atomic_write(parties_df, parties_path)
    _atomic_write(attorneys_df, attorneys_path)
    _atomic_write(judges_df, judges_path)
    _atomic_write(edges_df, edges_path)

    return {
        "dockets_path": str(dockets_path),
        "parties_path": str(parties_path),
        "attorneys_path": str(attorneys_path),
        "judges_path": str(judges_path),
        "edges_path": str(edges_path),
    }
