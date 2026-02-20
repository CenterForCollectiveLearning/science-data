import csv
import gzip
import json
import re
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


PREFIX_LEN = 22
MAIN_NAME = "main"
COMPLETE_FILTER = "all"
DN = "display_name"
IDC = "id"
PARID = "parent_id"
PUBY = "publication_year"
CC = "country_code"

DL_FRAME = "derive_links{}"
A2_GEN = "a2_init_atts"
COUNT_SUFFS = ["work-count", "cite-count"]

# app_root = Path(os.environ["RANKLESS_APP_ROOT"]) / "src/lib/assets/data"
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


class GenReader:
    def __init__(self, gen_root=".") -> None:
        self.root = gen_root
        self.counts_df = get_counts_df(gen_root)
        self.names_df = get_src_df(["names"], [A2_GEN], gen_root)
        self.gen_txt = "\n\n".join(
            fp.read_text() for fp in Path(f"{gen_root}/rankless_rs/src/gen/").iterdir()
        )

    def get_wcounts(self, etype) -> np.ndarray:
        return self._get_counts(etype, 0)

    def get_ccounts(self, etype) -> np.ndarray:
        return self._get_counts(etype, 1)

    def get_names(self, e: str):
        rex = rf"impl VariableSizeAttribute for {e.title()}Names {{ type SizeType = u(\d+); type LocType = .*; }}"
        sizet = int(re.findall(rex, self.gen_txt)[0])
        return get_str_arr(f"a2_init_atts/{e}-names", sizet)

    def _get_counts(self, etype, i):
        ind = (etype, COUNT_SUFFS[i])
        (dt, _, dl_name) = self.counts_df.loc[ind, :]
        dli = int(dl_name[-1])
        return get_dl_arr("-".join(ind), dli, dt[1:])


def get_flag_emoji(ccode: str):
    return (
        bytes([240, 159, 135, 101 + ord(ccode[0])])
        + bytes([240, 159, 135, 101 + ord(ccode[1])])
    ).decode("utf-8")


def get_root(entity):
    return oa_root / StowC.entity_csvs / entity


def get_csv_path(entity: str, sub: str):
    return get_root(entity) / f"{sub}.csv.gz"


def get_gen_dl(n, gen_root="."):
    return get_gen_src(DL_FRAME.format(n), gen_root)


def get_gen_src(suff, gen_root="."):
    return Path(f"{gen_root}/rankless_rs/src/gen/{suff}.rs").read_text()


def get_arr(slug, dt):
    blob = (oa_root / slug).read_bytes()
    return np.frombuffer(blob, np_dtype(dt))


def get_str_arr(slug: str, sizet: int):
    ss = 0
    names = []
    ftxt = (oa_root / f"{slug}/targets").read_bytes()
    for s in map(int, get_arr(f"{slug}/sizes", sizet)):
        names.append(ftxt[ss : (ss + s)].decode())
        ss = ss + s
    return names[:-1]


def np_dtype(n):
    return np.dtype(f"uint{n}").newbyteorder(">")


def get_dl_arr(ename, dli, dt):
    return get_arr(f"derive_links{dli}/{ename}", dt)


def get_counts_df(gen_root="."):
    return get_src_df(COUNT_SUFFS, [DL_FRAME.format(i) for i in [3, 4, 5]], gen_root)


def get_src_df(type_suffs, gen_files, gen_root="."):
    dfs = []
    for suff in type_suffs:
        rex = get_rex(suff)
        for gfile in gen_files:
            _df = pd.DataFrame(rex.findall(get_gen_src(gfile, gen_root))).assign(
                suff=suff, gfile=gfile
            )
            dfs.append(_df)
    return pd.concat(dfs, ignore_index=True).set_index([2, "suff"])


def get_rex(suff):
    return re.compile(
        rf'impl Entity for .* {{ type T = (.*); const N: usize = (\d+); const NAME: & str = "(.*)\-{suff}"; }}'
    )


def get_last_filter(entity):
    diriter = Path(oa_root, StowC.filter_steps).iterdir()
    went = filter(lambda p: entity in map(lambda sp: sp.name, p.iterdir()), diriter)
    p = sorted(went, key=lambda p: int(p.name))[-1] / entity
    return np.frombuffer(p.read_bytes(), dtype=np.dtype(np.uint64).newbyteorder(">"))


def get_filtered_main_df(ent: str, cols=None) -> pd.DataFrame:
    laf = get_last_filter(ent)
    return pd.concat(
        map(
            lambda df: df.assign(id=df[IDC].pipe(parse_id)).loc[
                lambda _df: _df[IDC].isin(laf), :
            ],
            iter_dfs(ent, cols=cols),
        )
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
    return col.str[PREFIX_LEN:].astype(np.uint64)


def id_from_s(s: str):
    return int(s[PREFIX_LEN:])


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
