"""
Driver page extractor (Motor Sport Magazine database) -> JSONL.
Regex + stdlib only (html.unescape just decodes entities), no HTML parsers.

Input:  data/html/database_drivers_<slug>.html
Incremental: existing records in the output files are kept; a driver found in the HTML is
overwritten if it is already there (its series rows are replaced too), otherwise appended.
Use --rebuild to build the files from scratch. Statistics and problems describe the current run only.

Output: data/drivers.jsonl           one line per driver
        data/driver_series.jsonl     one line per (driver, series)
        data/extract_stats.json      extraction statistics
        data/extract_problems.jsonl  every problem found (one line per problem)
"""
import argparse
import datetime
import glob
import html
import json
import os
import re
from collections import Counter

# ---------- regexes ----------
FLAGS = re.DOTALL

CANONICAL_RE = re.compile(r'<link rel="canonical" href="(https?://[^/"]+/database/drivers/([a-z0-9\-]+)/)"')
NAME_RE = re.compile(r'<h1 class="hero__title[^"]*">(.*?)</h1>', FLAGS)
TAGLINE_RE = re.compile(r'<p class="hero__text">(.*?)</p>', FLAGS)
# some pages have the canonical in the form /database/?post_type=driver&p=ID -> take the slug from the oEmbed link
OEMBED_RE = re.compile(r'rel="alternate"[^>]*oembed[^>]*%2Fdatabase%2Fdrivers%2F([a-z0-9\-]+)%2F')
SITE_DRIVER_URL = "https://www.motorsportmagazine.com/database/drivers/{}/"
GENDER_RE = re.compile(r'<main class="[^"]*\bgender-(\w+)')
WP_ID_RE = re.compile(r'data-behaviour="driver-data-box" data-driver-id="(\d+)"')

# the data box with the driver's vital data and per-series statistics
BOX_RE = re.compile(r'data-behaviour="driver-data-box"[^>]*>(.*?)(?=database-data__expand)', FLAGS)

# generic "label -> value" field inside the data box ({label} is filled in by simple_field)
SIMPLE_FIELD = r'database-data__head">{label}</div><div class="database-data__content[^"]*">(.*?)</div>'
NATIONALITY_RE = re.compile(
    r'database-data__head">Nationality</div><div class="database-data__content[^"]*">'
    r'(?:<div class="flag[^>]*>.*?</div>)?([^<]*)</div>', FLAGS)
LAST_RACE_RE = re.compile(
    r'database-data__head">Recent Race</div><div class="database-data__content[^"]*">'
    r'<a href="[^"]*/races/([^/"]+)/">([^<]+)</a>', FLAGS)

# date like '2nd November 2005', 'November 2005' or '2005'
DATE_RE = re.compile(r'^\s*(?:(\d{1,2})(?:st|nd|rd|th)\s+)?(?:([A-Za-z]+)\s+)?(\d{4})\b')
CIRCA_RE = re.compile(r'^\s*(?:circa|ca\.?|c\.)(?=\s|\d)\s*', re.IGNORECASE)
UNKNOWN_RE = re.compile(r'^\s*(?:-\s*)?unknown\b', re.IGNORECASE)
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}

# series: the <option> list gives the names, the stats blocks give the numbers (both keyed by series id)
SERIES_OPTION_RE = re.compile(r'<option data-target="([^"]+)">([^<]+)</option>')
SERIES_BLOCK_RE = re.compile(
    r'<div data-target="([^"]+)" class="database-data__seasons-stats[^"]*">(.*?)'
    r'(?=<div data-target="[^"]+" class="database-data__seasons-stats|<div class="database-data__expand"|$)',
    FLAGS)
YEARS_RE = re.compile(
    r'Years in Database</div><div class="database-data__content[^"]*">(\d+)'
    r'(?:\s*\((\d{4})(?:\s*-\s*(\d{4}))?\))?', FLAGS)
NUM_FIELD = r'database-data__head">{label}</div><div class="database-data__content[^"]*">\s*(\d+)'

BIO_SECTION_RE = re.compile(r'<section class="c-expanding-content(.*?)expanding-content__buttons', FLAGS)
# without the 'Read full bio' button (short bio) the section ends at the first </section>
BIO_SECTION_NOBTN_RE = re.compile(r'<section class="c-expanding-content(.*?)</section>', FLAGS)
BIO_BLOCK_RE = re.compile(r'<(p|h3|h4)\b[^>]*>(.*?)</\1>', FLAGS)
TEAMS_SECTION_RE = re.compile(r'id="teams">(.*?)</section>', FLAGS)
TEAM_RE = re.compile(r'/database/teams/[^/"]+/">([^<]+)</a>')
TAG_RE = re.compile(r'<[^>]+>')


# ---------- helper functions ----------
def clean(text):
    """Strip tags, decode entities, normalize whitespace."""
    text = html.unescape(TAG_RE.sub("", text)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def simple_field(box, label):
    """Return the cleaned text of the data-box field with the given label, or None."""
    m = re.search(SIMPLE_FIELD.format(label=label), box, FLAGS)
    return clean(m.group(1)) if m else None


def num_field(block, label):
    """Return the integer value of a numeric field (Starts, Wins...) in a series block, or None."""
    m = re.search(NUM_FIELD.format(label=label), block, FLAGS)
    return int(m.group(1)) if m else None


def read_jsonl(path):
    """Yield one parsed JSON object per non-empty line; a missing file gives nothing."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    """Write one JSON object per line via a temporary file, so a crash cannot corrupt the output."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def parse_date(raw):
    """
    '2nd November 2005 - Athens' -> ('2005-11-02', 'full').
    The rest (age, place) is ignored. A missing day/month gives a partial date
    ('2005-11', '2005'). An invalid format -> (None, 'unparsed').
    No value -> (None, 'missing'); 'Unknown' -> (None, 'unknown');
    only a place without a date -> (None, 'place_only'). The output is null in these cases.
    """
    if not raw:
        return None, "missing"
    if UNKNOWN_RE.match(raw):
        return None, "unknown"          # 'Unknown' (also 'Unknown - place')
    cm = CIRCA_RE.match(raw)
    if cm:                              # 'circa 1950 - Paris' -> ('1950', 'circa')
        iso, prec = parse_date(raw[cm.end():])
        return (iso, "circa") if iso else (None, prec)
    if not re.search(r"\d", re.sub(r"\(.*?\)", "", raw)):
        return None, "place_only"       # only a place, no date ('- Athens')
    m = DATE_RE.match(raw)
    if not m:
        return None, "unparsed"
    day, month, year = m.groups()
    if month is not None and month not in MONTHS:
        return None, "unparsed"
    if day and not month:
        return None, "unparsed"
    if day:
        try:
            datetime.date(int(year), MONTHS[month], int(day))    # reject impossible dates (e.g. 31 February)
        except ValueError:
            return None, "unparsed"
        return f"{year}-{MONTHS[month]:02d}-{int(day):02d}", "full"
    if month:
        return f"{year}-{MONTHS[month]:02d}", "month"
    return year, "year"


AGE_RE = re.compile(r"\s*\(\s*Age[^)]*\)", re.IGNORECASE)


def vital_text(label, raw):
    """'Born: 2nd November 2005 - Athens'. The age '(Age N)' is left out, as are 'Unknown' and empty values."""
    if not raw or UNKNOWN_RE.match(raw):
        return None
    v = re.sub(r"^\s*-\s*", "", AGE_RE.sub("", raw))
    v = re.sub(r"\s+", " ", v).strip(" -")
    return f"{label}: {v}" if v else None


# ---------- extraction of a single page ----------
def extract_page(text):
    """Return (driver_dict, [series_dict]) or (None, []) if this is not a driver page."""
    m = CANONICAL_RE.search(text)
    n = NAME_RE.search(text)
    if m:
        url, slug = m.group(1), m.group(2)
    else:
        o = OEMBED_RE.search(text)
        slug = o.group(1) if o else None
        url = SITE_DRIVER_URL.format(slug) if slug else None
    if not slug or not n:
        return None, []

    box_m = BOX_RE.search(text)
    box = box_m.group(1) if box_m else ""

    wp = WP_ID_RE.search(text)
    gender = GENDER_RE.search(text)
    tagline = TAGLINE_RE.search(text)
    nat = NATIONALITY_RE.search(box)
    last = LAST_RACE_RE.search(box)

    born_raw = simple_field(box, "Born")
    died_raw = simple_field(box, "Died")
    born_date, born_prec = parse_date(born_raw)
    died_date, died_prec = parse_date(died_raw)

    # biography
    bio = None
    bs = BIO_SECTION_RE.search(text) or BIO_SECTION_NOBTN_RE.search(text)
    has_bio_section = bool(bs)
    if bs:
        parts = [clean(b.group(2)) for b in BIO_BLOCK_RE.finditer(bs.group(1))
                 if "product-card" not in b.group(0)[:120]]       # skip embedded product cards
        parts = [p for p in parts if p]
        if parts:
            bio = re.sub(r"\s+", " ", " ".join(parts)).strip()   # no newlines

    # prepend the full born/died data (including the place) to the bio text so it is searchable
    real_bio = bio is not None
    head = [t for t in (vital_text("Born", born_raw), vital_text("Died", died_raw)) if t]
    if head:
        bio = ". ".join(head) + "." + (" " + bio if bio else "")

    # teams
    teams = []
    ts = TEAMS_SECTION_RE.search(text)
    if ts:
        teams = [clean(t) for t in TEAM_RE.findall(ts.group(1))]

    driver = {
        "driver_id": slug,
        "wp_id": int(wp.group(1)) if wp else None,
        "url": url,
        "name": clean(n.group(1)),
        "full_name": simple_field(box, "Full name"),
        "gender": gender.group(1) if gender else None,
        "nationality": clean(nat.group(1)) if nat and clean(nat.group(1)) else None,
        "born_date": born_date,
        "died_date": died_date,
        "last_race_slug": last.group(1) if last else None,
        "last_race_name": clean(last.group(2)) if last else None,
        "tagline": clean(tagline.group(1)) if tagline and clean(tagline.group(1)) else None,
        "bio": bio,
        "teams": teams,
        "_born_prec": born_prec,   # statistics only, removed before writing
        "_died_prec": died_prec,
        "_has_bio_section": has_bio_section,
        "_real_bio": real_bio,
        "_born_raw": born_raw,
        "_died_raw": died_raw,
    }

    # series
    names = {sid: clean(name) for sid, name in SERIES_OPTION_RE.findall(box)}
    series = []
    for sid, block in SERIES_BLOCK_RE.findall(box):
        y = YEARS_RE.search(block)
        years_count = int(y.group(1)) if y else None
        year_from = int(y.group(2)) if y and y.group(2) else None
        year_to = int(y.group(3)) if y and y.group(3) else year_from
        series.append({
            "driver_id": slug,
            "series_id": sid,
            "series_name": names.get(sid),
            "years_count": years_count,
            "year_from": year_from,
            "year_to": year_to,
            "starts": num_field(block, "Starts"),
            "wins": num_field(block, "Wins"),
            "poles": num_field(block, "Poles"),
            "podiums": num_field(block, "Podiums"),
        })
    return driver, series


# ---------- main run ----------
def main():
    ap = argparse.ArgumentParser(description="Driver page extractor -> JSONL")
    ap.add_argument("--input", default="data/html", help="folder with the HTML files")
    ap.add_argument("--out", default="data", help="output folder")
    ap.add_argument("--pattern", default="*.html", help="glob pattern of the input files")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore the existing JSONL files and build them from scratch "
                         "(by default existing records are kept and only updated/added)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, args.pattern)))
    os.makedirs(args.out, exist_ok=True)
    drivers_path = os.path.join(args.out, "drivers.jsonl")
    series_path = os.path.join(args.out, "driver_series.jsonl")

    # existing data (incremental mode): driver_id -> record, driver_id -> [series rows]
    drivers_db, series_db = {}, {}
    if not args.rebuild:
        drivers_db = {d["driver_id"]: d for d in read_jsonl(drivers_path)}
        for s in read_jsonl(series_path):
            series_db.setdefault(s["driver_id"], []).append(s)
        print(f"Loaded existing data: {len(drivers_db)} drivers, "
              f"{sum(len(v) for v in series_db.values())} series rows.")

    st = Counter()                 # extraction statistics (of this run only)
    series_counter = Counter()     # number of drivers per series id
    problems = []
    seen = set()                   # driver ids already processed in this run (duplicate protection)

    for path in files:
        st["files"] += 1
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        drv, series = extract_page(text)
        if drv is None:
            st["skipped_not_driver_page"] += 1
            problems.append({"file": os.path.basename(path), "issue": "not_driver_page"})
            continue
        if drv["driver_id"] in seen:
            st["skipped_duplicate"] += 1
            continue
        seen.add(drv["driver_id"])

        st["drivers"] += 1
        st["updated_existing" if drv["driver_id"] in drivers_db else "added_new"] += 1
        st["with_bio" if drv["_real_bio"] else "without_bio"] += 1   # only a real bio, not the born/died text
        st["with_born_died_text"] += 1 if drv["bio"] and not drv["_real_bio"] else 0
        st["with_series" if series else "without_series"] += 1
        st["with_teams" if drv["teams"] else "without_teams"] += 1
        st["with_nationality" if drv["nationality"] else "without_nationality"] += 1
        st["with_last_race" if drv["last_race_slug"] else "without_last_race"] += 1
        st["with_full_name" if drv["full_name"] else "without_full_name"] += 1
        st[f"born_{drv['_born_prec']}"] += 1
        st[f"died_{drv['_died_prec']}"] += 1
        for label in ("born", "died"):
            if drv[f"_{label}_prec"] == "unparsed":
                problems.append({"file": os.path.basename(path), "driver_id": drv["driver_id"],
                                 "issue": f"unparsed_{label}", "raw": drv[f"_{label}_raw"]})
        if drv["died_date"] and not drv["born_date"]:
            st["died_without_born"] += 1
        if drv["born_date"] and drv["died_date"] and drv["died_date"] < drv["born_date"]:
            st["died_before_born"] += 1
            problems.append({"file": os.path.basename(path), "driver_id": drv["driver_id"],
                             "issue": "died_before_born",
                             "raw": f"{drv['born_date']} / {drv['died_date']}"})
        for s in series:
            series_counter[s["series_id"]] += 1
            if None in (s["starts"], s["wins"], s["poles"], s["podiums"]):
                st["series_rows_with_null_stat"] += 1
        st["series_rows"] += len(series)

        # remove the helper fields (they were only needed for the statistics above)
        drv.pop("_born_prec")
        drv.pop("_died_prec")
        if drv.pop("_has_bio_section") and not drv["_real_bio"]:
            st["bio_section_but_empty"] += 1
            problems.append({"file": os.path.basename(path), "driver_id": drv["driver_id"],
                             "issue": "bio_section_but_empty"})
        drv.pop("_real_bio")
        drv.pop("_born_raw")
        drv.pop("_died_raw")
        # an existing driver is overwritten in place (keeps its position), a new one is appended
        drivers_db[drv["driver_id"]] = drv
        series_db[drv["driver_id"]] = series     # the old series rows of this driver are replaced

    # write the merged data (atomically, so a crash cannot corrupt the existing files)
    write_jsonl(drivers_path, drivers_db.values())
    ordered = [s for did in drivers_db for s in series_db.get(did, [])]
    ordered += [s for did, rows in series_db.items() if did not in drivers_db for s in rows]   # orphan rows are kept
    write_jsonl(series_path, ordered)
    st["total_drivers_in_file"] = len(drivers_db)
    st["total_series_rows_in_file"] = len(ordered)

    n = st["drivers"] or 1
    stats = dict(st)
    stats["pct_with_bio"] = round(100 * st["with_bio"] / n, 2)
    stats["pct_with_series"] = round(100 * st["with_series"] / n, 2)
    stats["avg_series_per_driver_with_series"] = (
        round(st["series_rows"] / st["with_series"], 2) if st["with_series"] else 0)
    stats["top_series"] = series_counter.most_common(15)
    stats["distinct_series"] = len(series_counter)
    stats["problems_sample"] = problems[:30]
    stats["problems_total"] = len(problems)

    # full list of problems (with the file name) - one line per problem
    with open(os.path.join(args.out, "extract_problems.jsonl"), "w", encoding="utf-8") as f:
        for pr in problems:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "extract_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()