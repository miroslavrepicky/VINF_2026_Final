"""
Custom indexer: builds two on-disk inverted indexes from drivers.jsonl + driver_series.jsonl.
No IR libraries, no databases, no lemmatization. Stdlib only. Searching lives in searcher.py.

    python3 indexer.py --drivers data/drivers.jsonl --series data/driver_series.jsonl --out data/index

Output:
  <out>/drivers/   1 document = 1 driver          (fields: name, full_name, nationality, teams,
                                                    series, tagline, last_race, bio, gender)
  <out>/series/    1 document = 1 driver+series   (fields: name, full_name, nationality, series)

Every field has its OWN inverted index (own df, own idf, own vector lengths):
  postings_<field>.jsonl  {"t": term, "p": [[doc_id, tf], ...]}        tf = occurrences of the term in the field
  terms.tsv               field, term, df, idf_classic, idf_smooth     (all idf values in one place)
  docs.jsonl              document metadata + norms {field: vector length} + max_tf {field: max tf}
  filters.json            auxiliary indexes for filters (team, nationality, series, statistic ranges)
  meta.json               N, vocabulary sizes, field weights, stopword usage, formulas

idf (both from the document frequency df in the given field, N = number of documents):
  classic : idf = log10(N / df)
  smooth  : idf = log10(1 + N / df)
Field weights are NOT applied at build time, only at search time (see meta.json), so they
can be changed without rebuilding the index.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

from tokenizer import normalize, tokenize

# field weight at search time: document score = sum of weight_of_field * score_in_field
DRIVER_WEIGHTS = {
    "name": 5.0, "full_name": 3.0, "nationality": 3.0, "teams": 2.0,
    "series": 2.0, "tagline": 2.0, "last_race": 1.0, "bio": 1.0, "gender": 2.0,
}
SERIES_WEIGHTS = {"name": 5.0, "full_name": 3.0, "nationality": 3.0, "series": 5.0, "gender": 2.0}
# gender is indexed as text together with common words, so a query like "female" / "women" finds women
GENDER_TEXT = {"male": "male man men", "female": "female woman women"}
STATS = ("starts", "wins", "poles", "podiums")


def read_jsonl(path):
    """Yield one parsed JSON object per non-empty line of a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def idf_classic(n, df):
    """Classic idf: log10(N / df)."""
    return math.log10(n / df)


def idf_smooth(n, df):
    """Smoothed idf: log10(1 + N / df); always positive, even for terms present in every document."""
    return math.log10(1 + n / df)


def series_stats(row):
    """Pick the statistics and year range of one driver_series row."""
    return {k: row.get(k) for k in (*STATS, "years_count", "year_from", "year_to")}


def career_totals(rows):
    """Sum of the statistics across all series. If no series has a value, the total is null."""
    out = {}
    for stat in STATS:
        vals = [r[stat] for r in rows if r.get(stat) is not None]
        out[stat] = sum(vals) if vals else None
    return out


def field_counters(fields, use_stopwords):
    """{field: Counter(term -> tf)}; empty fields are left out."""
    out = {}
    for field, text in fields.items():
        c = Counter(tokenize(text, use_stopwords))
        if c:
            out[field] = c
    return out


def build_filters(docs):
    """
    Auxiliary inverted indexes for filters (so filtering does not have to scan all documents):
      teams / series_tokens : word -> [doc_id]
      nationality / series_id / gender : value -> [doc_id]
      stats : "<career|series_id>.<stat>" -> [[value, doc_id], ...] sorted by value
              (a range query > >= < <= = is answered with binary search)
    """
    teams, nat, sid_idx, stok, gender = (defaultdict(list) for _ in range(5))
    stats = defaultdict(list)
    for d in docs:
        i = d["id"]
        for tok in sorted({t for team in d["teams"] for t in tokenize(team, False)}):
            teams[tok].append(i)
        if d["nationality"]:
            nat[normalize(d["nationality"]).strip()].append(i)
        if d.get("gender"):
            gender[d["gender"].lower()].append(i)
        for sid in d["series_names"]:
            sid_idx[sid].append(i)
        for tok in sorted({t for n in d["series_names"].values() for t in tokenize(n, False)}):
            stok[tok].append(i)
        for stat in STATS:
            if d["career"].get(stat) is not None:
                stats[f"career.{stat}"].append((d["career"][stat], i))
            for sid, row in d["series"].items():
                if row.get(stat) is not None:
                    stats[f"{sid}.{stat}"].append((row[stat], i))
    for k in stats:
        stats[k].sort()
    return {"teams": teams, "nationality": nat, "gender": gender, "series_id": sid_idx,
            "series_tokens": stok, "stats": stats}


def write_index(entries, weights, out_dir, use_stopwords):
    """entries: [(doc_meta_dict, {field: Counter})] -> index files in out_dir. Returns the meta dict."""
    n = len(entries)
    os.makedirs(out_dir, exist_ok=True)
    # per-field postings: field -> term -> [(doc_id, tf), ...]
    postings = {f: defaultdict(list) for f in weights}
    for doc_id, (_, fields) in enumerate(entries):
        for field, tf in fields.items():
            for term, c in tf.items():
                postings[field][term].append((doc_id, c))

    # vector lengths (cosine, idf_classic) and max tf per field
    norm_sq = {f: [0.0] * n for f in weights}
    for field, post in postings.items():
        for plist in post.values():
            idf = idf_classic(n, len(plist))
            for d, tf in plist:
                w = (1 + math.log10(tf)) * idf
                norm_sq[field][d] += w * w

    docs = []
    for doc_id, (meta, fields) in enumerate(entries):
        docs.append({
            "id": doc_id, **meta,
            "norms": {f: math.sqrt(norm_sq[f][doc_id]) for f in fields},
            "max_tf": {f: max(tf.values()) for f, tf in fields.items()},
        })
    with open(os.path.join(out_dir, "docs.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    vocab = {}
    with open(os.path.join(out_dir, "terms.tsv"), "w", encoding="utf-8") as tf_out:
        tf_out.write("field\tterm\tdf\tidf_classic\tidf_smooth\n")
        for field in weights:
            post = postings[field]
            vocab[field] = len(post)
            with open(os.path.join(out_dir, f"postings_{field}.jsonl"), "w", encoding="utf-8") as f:
                for term in sorted(post):
                    f.write(json.dumps({"t": term, "p": post[term]}, ensure_ascii=False) + "\n")
                    df = len(post[term])
                    tf_out.write(f"{field}\t{term}\t{df}\t{idf_classic(n, df)!r}\t{idf_smooth(n, df)!r}\n")

    with open(os.path.join(out_dir, "filters.json"), "w", encoding="utf-8") as f:
        json.dump(build_filters(docs), f, ensure_ascii=False)
    meta = {"N": n, "vocabulary": vocab, "field_weights": weights, "stopwords": use_stopwords,
            "idf_classic": "log10(N/df)", "idf_smooth": "log10(1+N/df)"}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Custom indexer (per-field inverted index, TF)")
    ap.add_argument("--drivers", default="data/drivers.jsonl", help="input file with drivers")
    ap.add_argument("--series", default="data/driver_series.jsonl", help="input file with per-series statistics")
    ap.add_argument("--out", default="data/index", help="output folder of the index")
    ap.add_argument("--no-stopwords", action="store_true", help="do not remove stopwords while tokenizing")
    args = ap.parse_args()
    use_sw = not args.no_stopwords

    # group the per-series rows by driver
    rows_by_driver = defaultdict(list)
    if os.path.exists(args.series):
        for s in read_jsonl(args.series):
            rows_by_driver[s["driver_id"]].append(s)

    driver_entries, series_entries = [], []
    for drv in read_jsonl(args.drivers):
        rows = rows_by_driver.get(drv["driver_id"], [])
        names = {r["series_id"]: (r.get("series_name") or r["series_id"]) for r in rows}
        base = {
            "driver_id": drv["driver_id"], "name": drv.get("name"), "url": drv.get("url"),
            "nationality": drv.get("nationality"), "gender": drv.get("gender"),
            "born_date": drv.get("born_date"),
            "died_date": drv.get("died_date"), "teams": drv.get("teams") or [],
        }

        # --- driver-level document ---
        fields = {
            "name": drv.get("name") or "", "full_name": drv.get("full_name") or "",
            "nationality": drv.get("nationality") or "",
            "teams": " ".join(drv.get("teams") or []), "series": " ".join(names.values()),
            "tagline": drv.get("tagline") or "", "last_race": drv.get("last_race_name") or "",
            "bio": drv.get("bio") or "",
            "gender": GENDER_TEXT.get((drv.get("gender") or "").lower(), ""),
        }
        driver_entries.append((
            {**base, "series_names": names,
             "series": {r["series_id"]: series_stats(r) for r in rows},
             "career": career_totals(rows)},
            field_counters(fields, use_sw)))

        # --- (driver, series)-level documents ---
        for r in rows:
            sid = r["series_id"]
            fields = {
                "name": drv.get("name") or "", "full_name": drv.get("full_name") or "",
                "nationality": drv.get("nationality") or "", "series": names[sid],
                "gender": GENDER_TEXT.get((drv.get("gender") or "").lower(), ""),
            }
            stats = series_stats(r)
            series_entries.append((
                {**base, "driver_series_id": f"{drv['driver_id']}/{sid}",
                 "series_names": {sid: names[sid]}, "series": {sid: stats},
                 "career": {k: stats[k] for k in STATS}},
                field_counters(fields, use_sw)))

    if not driver_entries:
        sys.exit("No documents to index.")

    out = {"drivers": write_index(driver_entries, DRIVER_WEIGHTS, os.path.join(args.out, "drivers"), use_sw)}
    if series_entries:
        out["series"] = write_index(series_entries, SERIES_WEIGHTS, os.path.join(args.out, "series"), use_sw)
    print(json.dumps({k: {"N": v["N"], "vocabulary": v["vocabulary"]} for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()