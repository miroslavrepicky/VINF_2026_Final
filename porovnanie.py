"""
Comparison of the two ranking methods (cosine / idf_classic vs smooth / idf_smooth) on several queries.
Writes a ready-to-paste DokuWiki text: for every query a table of idf values of its terms and a
side-by-side table of the top results of both methods.

    python3 porovnanie.py                                   (3 default queries, top 5 -> data/porovnanie.txt)
    python3 porovnanie.py --queries "senna" "british ferrari monaco" "hill"
    python3 porovnanie.py --top 10 --level series

Only free-text queries make sense here (filters do not use the ranking).
"""
import argparse
import os

from searcher import Index

DEFAULT_QUERIES = ["senna", "british champion", "monaco ferrari win"]   # rare term / common words / mix


def idf_rows(idx, units):
    """[(field, term, df, idf_classic, idf_smooth)] for the query terms found in the index."""
    rows = []
    for fld, term in units:
        fields = idx.weights if fld == "*" else [fld]
        for f in fields:
            if term in idx.terms.get(f, {}):
                df, ic, ism = idx.terms[f][term]
                rows.append((f, term, df, ic, ism))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Compare cosine and smooth ranking on several queries (DokuWiki output)")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--level", choices=["driver", "series"], default="driver")
    ap.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="data/porovnanie.txt")
    args = ap.parse_args()

    idx = Index(os.path.join(args.index, "drivers" if args.level == "driver" else "series"))
    out = [f"Index: {args.level}, N = {idx.N} dokumentov.\n"]

    for n, q in enumerate(args.queries, 1):
        groups, _, _ = idx.parse(q)
        units = list(dict.fromkeys(u for g in groups for u in g))
        out.append(f"=== Dopyt {n}: ''{q}'' ===\n")

        out.append("Hodnoty idf termov dopytu (df je počet dokumentov, v ktorých sa term v poli vyskytuje):\n")
        out.append("^ Pole ^ Term ^ df ^ idf_classic ^ idf_smooth ^")
        rows = idf_rows(idx, units)
        for f, t, df, ic, ism in rows:
            out.append(f"| {f} | {t} | {df} | {ic:.4f} | {ism:.4f} |")
        if not rows:
            out.append("| (žiadny term dopytu nie je v indexe) | | | | |")
        out.append("")

        res = {m: idx.search(q, m, args.top) for m in ("cosine", "smooth")}
        ids = {m: [d["driver_series_id" if args.level == "series" else "driver_id"] for d, _, _ in r]
               for m, r in res.items()}
        out.append(f"Prvých {args.top} výsledkov oboch metód:\n")
        out.append("^ Poradie ^ cosine ^ skóre ^ smooth ^ skóre ^")
        for i in range(args.top):
            cells = []
            for m in ("cosine", "smooth"):
                if i < len(res[m]):
                    d, s, _ = res[m][i]
                    label = d["name"] if args.level == "driver" else f"{d['name']} ({list(d['series_names'].values())[0]})"
                    cells += [label, f"{s:.4f}"]
                else:
                    cells += ["", ""]
            out.append(f"| {i + 1} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
        common = set(ids["cosine"]) & set(ids["smooth"])
        same_order = sum(1 for a, b in zip(ids["cosine"], ids["smooth"]) if a == b)
        out.append(f"\nSpoločných výsledkov v prvých {args.top}: {len(common)}, na rovnakej pozícii: {same_order}.\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()