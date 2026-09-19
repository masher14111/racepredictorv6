"""Left-join results enrichment onto the Betfair SP backbone."""
import pandas as pd

from utils.logger import get_logger
from utils.text_norm import norm_horse as _norm_horse
from utils.text_norm import norm_venue as _norm_venue
from utils.text_norm import minute_key as _minute_key

logger = get_logger(__name__)

ENRICH_COLS = ["jockey_id", "jockey_name", "trainer_id", "trainer_name",
               "position", "going", "result_source"]


def _key(venue, horse, when) -> tuple:
    return (_norm_venue(venue), _norm_horse(horse), _minute_key(when))


def join(backbone: pd.DataFrame, result_rows: list, priority: list) -> pd.DataFrame:
    """Return backbone with ENRICH_COLS filled from result_rows (left join)."""
    out = backbone.copy()
    for col in ENRICH_COLS:
        out[col] = pd.NA

    rank = {name: i for i, name in enumerate(priority)}
    best: dict = {}
    for r in result_rows:
        k = _key(r.venue, r.horse_name, r.race_date)
        cur = best.get(k)
        r_rank = rank.get(r.source, len(priority))
        if cur is None or r_rank < cur[0]:
            best[k] = (r_rank, r)
        elif r_rank != cur[0]:
            continue
        else:
            if cur[1].position is not None and r.position is not None \
                    and cur[1].position != r.position:
                logger.warning("data_quality: position conflict for %s", k)

    for idx, row in out.iterrows():
        k = _key(row["venue"], row["horse_name"], row["race_date"])
        hit = best.get(k)
        if not hit:
            continue
        r = hit[1]
        out.at[idx, "jockey_id"] = r.jockey_id
        out.at[idx, "jockey_name"] = r.jockey_name
        out.at[idx, "trainer_id"] = r.trainer_id
        out.at[idx, "trainer_name"] = r.trainer_name
        out.at[idx, "position"] = r.position
        out.at[idx, "going"] = r.going
        out.at[idx, "result_source"] = r.source
    return out
