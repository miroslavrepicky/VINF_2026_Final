"""
Pick the N longest driver records from drivers.jsonl together with their rows from
driver_series.jsonl and write them out as ready-to-paste examples for the wiki.

    python3 ukazky.py                                  (5 longest -> data/ukazky.txt, DokuWiki format)
    python3 ukazky.py --n 3 --by bio --bio-chars 0     (3 longest bios, bio is NOT shortened)
    python3 ukazky.py --format json                    (plain JSON instead of DokuWiki)

"Longest" = the longest serialized record (--by record, default) or the longest bio text (--by bio).
"""
import argparse
import json
from collections import defaultdict


def read_jsonl(path):
    """Yield one parsed JSON object per non-empty line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def shorten_bio(drv, limit):
    """Return a copy of the driver with the bio cut to `limit` characters (0 = no cutting)."""
    drv = dict(drv)
    bio = drv.get("bio")
    if limit and bio and len(bio) > limit:
        drv["bio"] = bio[:limit].rstrip() + " [...]"
    return drv


def main():
    ap = argparse.ArgumentParser(description="Examples of the longest driver records for the documentation")
    ap.add_argument("--drivers", default="data/drivers.jsonl")
    ap.add_argument("--series", default="data/driver_series.jsonl")
    ap.add_argument("--n", type=int, default=5, help="number of examples")
    ap.add_argument("--by", choices=["record", "bio"], default="record",
                    help="what 'longest' means: whole serialized record or the bio text")
    ap.add_argument("--bio-chars", type=int, default=400,
                    help="cut the bio in the output to this many characters (0 = do not cut)")
    ap.add_argument("--format", choices=["dokuwiki", "json"], default="dokuwiki")
    ap.add_argument("--out", default="data/ukazky.txt")
    args = ap.parse_args()

    drivers = list(read_jsonl(args.drivers))
    rows = defaultdict(list)
    for s in read_jsonl(args.series):
        rows[s["driver_id"]].append(s)

    def size(d):
        return len(d.get("bio") or "") if args.by == "bio" else len(json.dumps(d, ensure_ascii=False))

    top = sorted(drivers, key=size, reverse=True)[:args.n]

    out = []
    for i, drv in enumerate(top, 1):
        drv_out = shorten_bio(drv, args.bio_chars)
        drv_json = json.dumps(drv_out, ensure_ascii=False, indent=2)
        series = rows.get(drv["driver_id"], [])
        series_json = "\n".join(json.dumps(s, ensure_ascii=False) for s in series) or "(no series rows)"
        if args.format == "dokuwiki":
            link = f" ([[{drv['url']}]])" if drv.get("url") else ""     # page of the driver next to the name
            out.append(f"==== Ukážka {i}: {drv['name']}{link} ====\n\n"
                       f"Záznam v ''drivers.jsonl'':\n\n<code json>\n{drv_json}\n</code>\n\n"
                       f"Zodpovedajúce riadky v ''driver_series.jsonl'' ({len(series)}):\n\n"
                       f"<code json>\n{series_json}\n</code>\n")
        else:
            out.append(f"# Example {i}: {drv['name']} ({drv.get('url') or 'no url'})\n{drv_json}\n# driver_series.jsonl ({len(series)} rows)\n{series_json}\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Written {len(top)} examples to {args.out}")


if __name__ == "__main__":
    main()