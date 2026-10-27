"""
Search over the index built by indexer.py (the whole index is loaded into memory).

    python3 searcher.py "team:ferrari wins>20" --index data/index
    python3 searcher.py "series:f2 wins>=3" --level series
    python3 searcher.py "name:hill bio:monaco" --index data/index
    python3 searcher.py --index data/index                       (interactive mode)

By default the results of BOTH ranking methods are printed for a query (--method cosine|smooth|both).

Levels (--level):
  driver (default)  result = a driver; statistics = career totals (sum over all series)
  series            result = a driver in one series (e.g. Graham Hill / F2) with the statistics of that series

Query syntax (combinable):
  ferrari monaco       free text: searched in ALL fields, score = sum of field_weight * score_in_field.
                       A match of at least one word is enough (soft "or"); the score is multiplied by the
                       query coverage = (number of query words the document hit) / (number of query words)
  french AND lourdes   operators AND / OR (UPPERCASE only): a document must satisfy at least one
  french OR greek      group separated by OR, and all words in a group must match (AND, space = AND).
  french AND lourdes OR greek AND athens      AND binds tighter than OR; there are no parentheses.
                       With AND/OR the coverage is not applied, the score is just the sum over fields.
  name:hill            search in a single field only (name, fullname, tagline, lastrace, bio); name:"graham hill"
  team:ferrari         filter: drove for a team whose name contains all the words (team:"red bull")
  nationality:british  filter: nationality (exact match)
  gender:female        filter: gender (male|female, also men/women)
  series:f2            filter: series by id (f2, f1, indycars) or by words of its name (series:"international f2")
  wins>20              filter on a statistic (starts|wins|poles|podiums; > >= < <= =):
                         without series: career total (at the series level: the statistic of that series)
                         with series:    the statistic in that series (all conditions on a single series row)
  f2.wins>=3           statistic in a specific series by id, independent of series:
A query with filters only is sorted by wins.

Ranking methods (both use the IDF from df in the given field, no probabilistic IDF).
Score of field f of document d for query q:
  cosine : w(t,d) = (1 + log10 tf) * idf_classic,  idf_classic = log10(N / df)
           score_f = sum_t w(t,q) * w(t,d) / (|q_f| * |d_f|)
  smooth : score_f = sum_t (0.5 + 0.5 * tf / max_tf_f(d)) * idf_smooth(t),  idf_smooth = log10(1 + N / df)
Final score = sum_f weight_f * score_f. The idf values are read from terms.tsv.
"""
import argparse
import json
import math
import os
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

from tokenizer import normalize, tokenize

# ---------- query syntax ----------
TEAM_RE = re.compile(r'team:(?:"([^"]+)"|(\S+))', re.IGNORECASE)
NAT_RE = re.compile(r'nationality:(?:"([^"]+)"|(\S+))', re.IGNORECASE)
SERIES_RE = re.compile(r'series:(?:"([^"]+)"|(\S+))', re.IGNORECASE)
STAT_RE = re.compile(
    r'(?:\b([a-z0-9][a-z0-9\-]*)\.)?\b(starts|wins|poles|podiums)\s*(>=|<=|>|<|=)\s*(\d+)', re.IGNORECASE)
GENDER_RE = re.compile(r'gender:(?:"([^"]+)"|(\S+))', re.IGNORECASE)
GENDER_ALIAS = {"male": "male", "m": "male", "man": "male", "men": "male",
                "female": "female", "f": "female", "woman": "female", "women": "female"}
FIELD_RE = re.compile(r'\b(name|fullname|tagline|lastrace|bio):(?:"([^"]+)"|(\S+))', re.IGNORECASE)
# query field name -> index field name
FIELD_ALIAS = {"name": "name", "fullname": "full_name", "tagline": "tagline",
               "lastrace": "last_race", "bio": "bio"}
OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "=": lambda a, b: a == b}
STAT_ORDER = ("starts", "wins", "poles", "podiums")


def read_jsonl(path):
    """Yield one parsed JSON object per non-empty line of a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# tokenizer of the free-text part of the query: operators, field:value, "phrases", plain words
ITEM_RE = re.compile(
    r'(?P<op>\b(?:AND|OR)\b)'
    r'|(?P<field>\b(?:name|fullname|tagline|lastrace|bio)):(?:"(?P<fq>[^"]+)"|(?P<fw>\S+))'
    r'|"(?P<phrase>[^"]+)"|(?P<word>\S+)', re.IGNORECASE)


def parse_query(query, use_stopwords=True):
    """
    Return (groups, has_operators, filters).
    groups: list of groups, a group = list of terms ("*", term) (searched in all fields)
            or (field, term) (searched only in that field). Groups are separated by OR, AND applies inside a group.
    Without AND/OR operators (uppercase) there is a single group and the mode is 'soft':
    a match of at least one term is enough, the score is multiplied by the query coverage.
    Filters (team:, nationality:, gender:, series:, wins>...) always apply as AND.
    """
    f = {"teams": [], "nationality": [], "gender": [], "series": [], "stats": []}
    for m in TEAM_RE.finditer(query):
        f["teams"].append(tokenize(m.group(1) or m.group(2), False))
    for m in NAT_RE.finditer(query):
        f["nationality"].append(normalize(m.group(1) or m.group(2)).strip())
    for m in GENDER_RE.finditer(query):
        f["gender"].append(GENDER_ALIAS.get((m.group(1) or m.group(2)).strip().lower(), "?"))
    for m in SERIES_RE.finditer(query):
        raw = (m.group(1) or m.group(2)).strip()
        f["series"].append((raw.lower(), tokenize(raw, False)))
    for m in STAT_RE.finditer(query):
        f["stats"].append((m.group(1).lower() if m.group(1) else None,
                           m.group(2).lower(), m.group(3), int(m.group(4))))
    # what remains after removing the filters is the free-text part
    text = query
    for rx in (STAT_RE, SERIES_RE, GENDER_RE, NAT_RE, TEAM_RE):
        text = rx.sub(" ", text)

    def build(stop):
        groups, cur, has_ops, words = [], [], False, 0
        for m in ITEM_RE.finditer(text):
            if m.group("op") and m.group("op") in ("AND", "OR"):       # operators only in uppercase
                has_ops = True
                if m.group("op") == "OR":
                    groups.append(cur)
                    cur = []
            elif m.group("field"):
                field = FIELD_ALIAS[m.group("field").lower()]
                words += 1
                cur += [(field, t) for t in tokenize(m.group("fq") or m.group("fw"), stop)]
            else:
                words += 1
                cur += [("*", t) for t in tokenize(m.group("phrase") or m.group("word") or m.group("op"), stop)]
        groups.append(cur)
        return [list(dict.fromkeys(g)) for g in groups if g], has_ops, words

    groups, has_ops, words = build(use_stopwords)
    if not groups and words:                    # the query consists only of stopwords
        groups, has_ops, _ = build(False)
    return groups, has_ops, f


def stats_ok(src, conds):
    """True if the statistics dict `src` satisfies all (stat, op, value) conditions."""
    for stat, op, val in conds:
        cur = src.get(stat) if src else None
        if cur is None or not OPS[op](cur, val):
            return False
    return True


class Index:
    """In-memory index of one level (drivers or series): postings, idf values, documents and filter indexes."""

    def __init__(self, path):
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.N = self.meta["N"]
        self.weights = self.meta["field_weights"]
        self.docs = list(read_jsonl(os.path.join(path, "docs.jsonl")))

        # per-field idf from terms.tsv: field -> term -> (df, idf_classic, idf_smooth)
        self.terms = {f: {} for f in self.weights}
        with open(os.path.join(path, "terms.tsv"), encoding="utf-8") as f:
            next(f)                                   # skip the header line
            for line in f:
                field, term, df, ic, ism = line.rstrip("\n").split("\t")
                self.terms[field][term] = (int(df), float(ic), float(ism))
        # per-field postings: field -> term -> [[doc_id, tf], ...]
        self.postings = {}
        for field in self.weights:
            self.postings[field] = {r["t"]: r["p"] for r in
                                    read_jsonl(os.path.join(path, f"postings_{field}.jsonl"))}
            for t, p in self.postings[field].items():
                if len(p) != self.terms[field][t][0]:
                    raise ValueError(f"df does not match the postings: {field}/{t}")

        self.team_tokens = [[set(tokenize(t, False)) for t in d["teams"]] for d in self.docs]
        self.series_ids = {sid for d in self.docs for sid in d["series"]}
        with open(os.path.join(path, "filters.json"), encoding="utf-8") as f:
            flt = json.load(f)
        self.f_teams, self.f_nat = flt["teams"], flt["nationality"]
        self.f_gender = flt["gender"]
        self.f_sid, self.f_stok = flt["series_id"], flt["series_tokens"]
        # statistics: key -> (sorted values, document ids)
        self.f_stats = {k: ([v for v, _ in rows], [i for _, i in rows])
                        for k, rows in flt["stats"].items()}
        self.last_candidates = None                   # number of filter candidates of the last search (for display)

    # ---- filters through the auxiliary indexes (filters.json) ----
    def stat_range(self, key, op, val):
        """Document ids whose statistic `key` satisfies `op val` (binary search in the sorted values)."""
        vals, ids = self.f_stats.get(key, ([], []))
        lo, hi = {">": (bisect_right(vals, val), len(vals)), ">=": (bisect_left(vals, val), len(vals)),
                  "<": (0, bisect_left(vals, val)), "<=": (0, bisect_right(vals, val)),
                  "=": (bisect_left(vals, val), bisect_right(vals, val))}[op]
        return set(ids[lo:hi])

    def candidates(self, f):
        """Set of doc_ids that may satisfy the filters (from the indexes), or None if the index cannot be used.
        The exact verification (team within a single name, statistics on a single series row) is done by evaluate()."""
        sets = []
        for want in f["teams"]:
            lists = [self.f_teams.get(t) for t in want]
            sets.append(set.intersection(*map(set, lists)) if want and all(l is not None for l in lists) else set())
        for nat in f["nationality"]:
            sets.append(set(self.f_nat.get(nat, [])))
        for g in f["gender"]:
            sets.append(set(self.f_gender.get(g, [])))
        for raw, toks in f["series"]:
            if raw in self.series_ids:
                sets.append(set(self.f_sid.get(raw, [])))
            else:
                lists = [self.f_stok.get(t) for t in toks]
                sets.append(set.intersection(*map(set, lists)) if toks and all(l is not None for l in lists) else set())
        for s, st, op, v in f["stats"]:
            if s:
                sets.append(self.stat_range(f"{s}.{st}", op, v))
            elif not f["series"]:
                sets.append(self.stat_range(f"career.{st}", op, v))
        if not sets:
            return None
        sets.sort(key=len)                            # intersect the smallest sets first
        return set.intersection(*sets)

    # ---- exact verification of the filters on the candidates ----
    def series_match(self, doc, sid, want):
        """True if the series row `sid` of the document matches the series filter (exact id first, then name words)."""
        raw, toks = want
        if raw in self.series_ids:                  # an exact id takes precedence
            return sid == raw
        return bool(toks) and set(toks) <= set(tokenize(doc["series_names"].get(sid, ""), False))

    def evaluate(self, d, f):
        """Return (ok, matching_series). The series are the ids of the rows on which the filter was satisfied."""
        doc = self.docs[d]
        for want in f["teams"]:
            if not want or not any(set(want) <= tt for tt in self.team_tokens[d]):
                return False, []
        for nat in f["nationality"]:
            if normalize(doc["nationality"] or "").strip() != nat:
                return False, []
        for g in f["gender"]:
            if (doc.get("gender") or "").lower() != g:
                return False, []
        scoped = [(s, st, op, v) for s, st, op, v in f["stats"] if s]       # e.g. f2.wins>=3
        plain = [(st, op, v) for s, st, op, v in f["stats"] if not s]       # e.g. wins>20
        for s, st, op, v in scoped:
            if not stats_ok(doc["series"].get(s), [(st, op, v)]):
                return False, []
        rows = list(doc["series"])
        for want in f["series"]:
            rows = [sid for sid in rows if self.series_match(doc, sid, want)]
        if f["series"]:
            if not rows:
                return False, []
            if plain:
                rows = [sid for sid in rows if stats_ok(doc["series"][sid], plain)]
                if not rows:
                    return False, []
            return True, rows
        if plain and not stats_ok(doc["career"], plain):
            return False, []
        return True, []

    # ---- per-field ranking ----
    def score_field(self, field, terms, method):
        """{doc_id: score} for one field; terms is a Counter(term -> tf in the query)."""
        known = {t: c for t, c in terms.items() if t in self.postings[field]}
        scores = defaultdict(float)
        if not known:
            return scores
        if method == "cosine":
            # query vector weights; the document weights are accumulated and then divided by both norms
            q_w = {t: (1 + math.log10(c)) * self.terms[field][t][1] for t, c in known.items()}
            q_norm = math.sqrt(sum(w * w for w in q_w.values()))
            if q_norm == 0:
                return scores
            for t, wq in q_w.items():
                idf = self.terms[field][t][1]
                for d, tf in self.postings[field][t]:
                    scores[d] += wq * (1 + math.log10(tf)) * idf
            for d in list(scores):
                denom = q_norm * self.docs[d]["norms"][field]
                scores[d] = scores[d] / denom if denom > 0 else 0.0
        elif method == "smooth":
            for t in known:
                idf = self.terms[field][t][2]
                for d, tf in self.postings[field][t]:
                    scores[d] += (0.5 + 0.5 * tf / self.docs[d]["max_tf"][field]) * idf
        else:
            raise ValueError(f"unknown method: {method}")
        return scores

    def parse(self, query):
        """Parse a query using the stopword setting the index was built with."""
        return parse_query(query, self.meta["stopwords"])

    def rank(self, units, method):
        """Weighted sum of the per-field scores. Terms ("*", t) go to all fields, (field, t) only to its own."""
        free = [t for fld, t in units if fld == "*"]
        by_field = defaultdict(list)
        for fld, t in units:
            if fld != "*":
                by_field[fld].append(t)
        total = defaultdict(float)
        for field, weight in self.weights.items():
            terms = free + by_field.get(field, [])
            if terms:
                for d, s in self.score_field(field, Counter(terms), method).items():
                    total[d] += weight * s
        return total

    def unit_docs(self, unit):
        """Documents in which the term matches (in any field, or in the given field)."""
        fld, t = unit
        fields = self.weights if fld == "*" else [fld]
        return {d for f_ in fields for d, _ in self.postings.get(f_, {}).get(t, ())}

    def boolean_docs(self, groups):
        """Union of the groups; within each group the intersection (AND) of the term document lists."""
        out = set()
        for g in groups:
            sets = sorted((self.unit_docs(u) for u in g), key=len)
            out |= set.intersection(*sets) if sets else set()
        return out

    def search(self, query, method="cosine", top=10, require_all=False):
        """Return [(doc, score or None, matching_series)]."""
        groups, has_ops, f = self.parse(query)
        units = list(dict.fromkeys(u for g in groups for u in g))
        has_text = bool(units)
        has_filters = any(f.values())
        cand = self.candidates(f) if has_filters else None
        self.last_candidates = None if cand is None else len(cand)
        if has_text:
            raw = self.rank(units, method)
            if has_ops:
                # explicit AND/OR: documents are selected by boolean logic, the score is unchanged
                scores = {d: raw.get(d, 0.0) for d in self.boolean_docs(groups)}
            else:
                # soft mode: score * (matched query terms / all query terms)
                counts = Counter(d for u in units for d in self.unit_docs(u))
                scores = {d: raw.get(d, 0.0) * c / len(units) for d, c in counts.items()
                          if raw.get(d, 0.0) > 0 and (not require_all or c == len(units))}
            if cand is not None:
                scores = {d: s for d, s in scores.items() if d in cand}
        elif has_filters:
            scores = {d: None for d in (cand if cand is not None else range(self.N))}   # filters only
        else:
            return []
        items = []
        for d, s in scores.items():
            ok, rows = self.evaluate(d, f)
            if ok:
                items.append((d, s, rows))
        if has_text:
            items.sort(key=lambda x: (-x[1], self.docs[x[0]]["name"] or ""))
        else:
            def wins(x):
                doc = self.docs[x[0]]
                src = [doc["series"][r].get("wins") or 0 for r in x[2]] or [doc["career"]["wins"] or 0]
                return -max(src)
            items.sort(key=lambda x: (wins(x), self.docs[x[0]]["name"] or ""))
        return [(self.docs[d], s, rows) for d, s, rows in items[:top]]


def fmt_stats(st):
    """Format the statistics as 'starts=.. wins=.. poles=.. podiums=..' ('?' for a missing value)."""
    return " ".join(f"{k}={st.get(k) if st.get(k) is not None else '?'}" for k in STAT_ORDER)


def show(results, level, as_json=False):
    """Print the results as text or (with as_json) as one JSON object per line."""
    if not results:
        print("(no results)")
        return
    for rank, (doc, score, rows) in enumerate(results, 1):
        shown = rows or (list(doc["series"]) if level == "series" else [])
        if as_json:
            print(json.dumps({"rank": rank, "score": None if score is None else round(score, 6),
                              "driver_id": doc["driver_id"], "name": doc["name"],
                              "nationality": doc["nationality"], "born_date": doc["born_date"],
                              "url": doc["url"], "career": doc["career"],
                              "series": {r: doc["series"][r] for r in shown}}, ensure_ascii=False))
            continue
        sc = f"{score:8.4f}" if score is not None else "       -"
        head = (f"{rank:>2}. {sc}  {doc['name']}  [{doc['nationality'] or '?'}, "
                f"{doc['born_date'] or '?'}]")
        if level == "series":
            sid = shown[0]
            st = doc["series"][sid]
            yrs = f"{st['year_from']}-{st['year_to']}" if st.get("year_from") else "?"
            print(f"{head}  {doc['series_names'][sid]} ({yrs})  {fmt_stats(st)}")
            print(f"{'':14}{doc['url']}")
        else:
            print(f"{head}  career: {fmt_stats(doc['career'])}")
            for sid in rows:
                print(f"{'':14}{doc['series_names'][sid]}: {fmt_stats(doc['series'][sid])}")
            print(f"{'':14}{doc['url']}")


def main():
    ap = argparse.ArgumentParser(description="Search the index (per-field TF-IDF)")
    ap.add_argument("query", nargs="?", default=None, help="query; without it the interactive mode starts")
    ap.add_argument("--index", default="data/index", help="index folder created by indexer.py")
    ap.add_argument("--level", choices=["driver", "series"], default="driver", help="result level")
    ap.add_argument("--method", choices=["cosine", "smooth", "both"], default="both", help="ranking method")
    ap.add_argument("--top", type=int, default=10, help="number of results")
    ap.add_argument("--all", action="store_true", help="a document must contain all query terms (without AND/OR in the query)")
    ap.add_argument("--json", action="store_true", help="print the results as JSON lines")
    args = ap.parse_args()

    idx = Index(os.path.join(args.index, "drivers" if args.level == "driver" else "series"))
    methods = ["cosine", "smooth"] if args.method == "both" else [args.method]

    def run(query):
        groups, _, _ = idx.parse(query)
        has_text = bool(groups)
        # without text there is no scoring, the methods would print the same output
        for m in (methods if has_text else methods[:1]):
            print(f"--- {m if has_text else 'filters only (sorted by wins)'} ---")
            show(idx.search(query, m, args.top, args.all), args.level, args.json)
            if idx.last_candidates is not None:
                print(f"(filters from the index: {idx.last_candidates} candidates out of {idx.N} documents)")

    if args.query:
        run(args.query)
    else:
        voc = sum(self_n for self_n in idx.meta["vocabulary"].values())
        print(f"Index ({args.level}): {idx.N} documents, {voc} terms across fields. Empty line = quit.")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q:
                break
            run(q)


if __name__ == "__main__":
    main()