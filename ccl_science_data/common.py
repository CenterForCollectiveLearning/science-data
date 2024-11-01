import csv
import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pandas.compat import os
from tqdm import tqdm

from .gen import AttC, ComC, EntC, Steps, StowC

print("loading ", find_dotenv())
load_dotenv()


MAIN_NAME = "main"
COMPLETE_FILTER = "all"
DN = "display_name"
IDC = "id"
PARID = "parent_id"
PUBY = "publication_year"

# app_root = Path(os.environ["RANKLESS_APP_ROOT"]) / "src/lib/assets/data"
oa_persistent = Path(os.environ["OA_PERSISTENT"])
oa_root = Path(os.environ["OA_ROOT"])
snap_dir = Path(os.environ["OA_SNAPSHOT"])

serve_path = oa_root / "pruned-cache"

print("OA_ROOT: ", oa_root)
print("SNAP_ROOT: ", snap_dir)


class Keys:
    ROOT = "root_entity_type"
    CITE = "citations"
    PAPER = "papers"
    SEM = "semantic_id"
    META = "meta"
    OA_ID_META = "oa_id"


class GzWriter:
    def __init__(self, out_dir: Path, name: str):
        self.fp = gzip.open(out_dir / f"{name}.csv.gz", "wt").__enter__()
        self.writer = None

    def append(self, d: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.fp, list(d.keys()))
            self.writer.writeheader()
        self.writer.writerow(d)

    def close(self, *exc):
        self.fp.__exit__(*exc)


def get_root(entity):
    return oa_root / StowC.entity_csvs / entity


def get_csv_path(entity: str, sub: str):
    return get_root(entity) / f"{sub}.csv.gz"


def get_last_filter(entity):
    diriter = Path(oa_root, StowC.filter_steps).iterdir()
    went = filter(lambda p: entity in map(lambda sp: sp.name, p.iterdir()), diriter)
    p = sorted(went, key=lambda p: int(p.name))[-1] / entity
    return np.frombuffer(p.read_bytes(), dtype=np.dtype(np.uint64).newbyteorder(">"))


def get_filtered_main_df(ent: str) -> pd.DataFrame:
    return (
        read_full_df(ent)
        .assign(id=lambda df: df["id"].pipe(parse_id))
        .loc[lambda df: df["id"].isin(get_last_filter(ent)), :]
    )


def iter_dfs(
    ent: str, sub: str = MAIN_NAME, chunk: int = 1_000_000, cols=None
) -> Iterable[pd.DataFrame]:
    for _df in pd.read_csv(get_csv_path(ent, sub), chunksize=chunk, usecols=cols):
        yield _df


def read_full_df(ent: str, sub: str = MAIN_NAME, cols=None):
    return pd.concat(iter_dfs(ent, sub, cols=cols))


def load_map(kind):
    blob = Path(oa_root, Steps.ENTITY_MAPPING, kind).read_bytes()
    imap = np.frombuffer(blob, dtype=np.dtype(np.uint64).newbyteorder(">")).reshape(
        -1, 2
    )
    return pd.DataFrame(imap).set_index(0).loc[:, 1].to_dict()


def parse_id(col):
    return col.str[22:].astype(np.uint64)


def read_p_gz(path: Path):
    return read_p(path.with_suffix(".json.gz"))


def read_p(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def dump_filtered(
    out_dir: Path,
    filter_set,
    main=EntC.WORKS,
    sub: str = MAIN_NAME,
    ic=IDC,
    cols=None,
    id_parser=parse_id,
):
    first_one = True
    out_path = out_dir / main / f"{sub}.csv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"")
    with gzip.open(out_path, "ab") as out_buf:
        for _df in tqdm(iter_dfs(main, sub, cols=cols), desc=sub):
            _df.loc[_df[ic].pipe(id_parser).isin(filter_set), :].to_csv(
                out_buf, index=False, header=first_one
            )
            first_one = False


def iter_snap_items(e: str, src_dir=snap_dir):
    rdir = src_dir / "data" / e
    jsfiles = []
    for subd in rdir.iterdir():
        if not subd.is_dir():
            continue
        jsfiles.extend(subd.iterdir())
    for jsf in tqdm(jsfiles, e):
        with gzip.open(jsf) as gzp:
            for gl in gzp:
                yield jsf, gl
