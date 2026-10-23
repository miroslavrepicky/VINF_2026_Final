import argparse
import asyncio
import json
import os
import re
from collections import deque
from urllib.parse import urljoin, urlparse, parse_qs
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

SITE = "https://www.motorsportmagazine.com"
BASE_URL = SITE + "/database/drivers/"
SEEDS = [BASE_URL]                               # štartovacie stránky (ďalšie cez --seed)
HOST = "www.motorsportmagazine.com"
ALLOW_EXTERNAL = False   # False = prehľadáva sa celá doména HOST (všetky cesty); True = aj odkazy na iné domény
KEEP_QUERY_PARAMS = ("page",)   # z query ostanú len tieto parametre (zvyšok by vyrábal nekonečné varianty URL)
SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf", ".zip", ".mp4", ".mp3",
                   ".css", ".js", ".json", ".xml", ".woff", ".woff2", ".ttf", ".avif", ".rss")
STATE_FILE = "crawl_state.json"                  # frontier + hotové + zlyhané (resume)
SAVE_DIR = "data/html"
HEADERS = {
    "User-Agent": "VINF-Project-Crawler/1.0 (student project at STU FIIT Bratislava, course VINF; contact: xrepickym@stuba.sk)"
}
SLEEP_TIME = 1.5
HEADLESS = False          # prepínač --headless; stránky jazdcov sa ťahali s oknom, tak je to predvolené
MAX_PAGES_PER_RUN = None  # stránok v jednom behu (None = bez limitu), prepíše --max-pages
MAX_RETRIES = 3           # po koľkých neúspešných pokusoch sa URL vzdá
SAVE_STATE_EVERY = 10     # ako často sa ukladá stav (okrem toho vždy pri ukončení)
NAV_TIMEOUT = 5000

DRIVER_PATH_RE = re.compile(r"^/database/drivers/[a-z0-9\-]+/$")
DB_DETAIL_RE = re.compile(r"^/database/[^/]+/[^/]+")
HREF_RE = re.compile(r'<a\s[^>]*?href=["\']([^"\'#]+)', re.I | re.S)


# ---------- súbory ----------

def atomic_write(path, text):
    """Zapíš do dočasného súboru a premenuj, aby pád uprostred zápisu nenechal poškodený súbor."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def html_path(url):
    filename = urlparse(url).path.strip("/").replace("/", "_") or "index"
    return f"{SAVE_DIR}/{filename}.html"


# ---------- URL: normalizácia a rozsah crawlu ----------

def normalize(base, href):
    """Vráti kanonickú URL, ak ju treba prehľadávať, inak None."""
    url = urljoin(base, href.strip().replace("&amp;", "&"))
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        return None                                  # mailto:, javascript:, tel: ...
    if not ALLOW_EXTERNAL and p.netloc != HOST:
        return None
    if p.path.lower().endswith(SKIP_EXTENSIONS):
        return None                                  # obrázky, pdf, css... nemá zmysel otvárať v prehliadači
    path = p.path or "/"
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"                                  # /x a /x/ je tá istá stránka (súbory s príponou sa nechajú)
    if p.netloc == HOST and DRIVER_PATH_RE.match(path):
        return f"{SITE}{path}"                       # stránka jazdca: bez query
    qs = parse_qs(p.query)
    kept = []
    for k in KEEP_QUERY_PARAMS:
        v = qs.get(k, [None])[0]
        if v and not (k == "page" and v == "1"):
            kept.append(f"{k}={v}")
    return f"{p.scheme}://{p.netloc}{path}" + ("?" + "&".join(kept) if kept else "")


def is_driver_url(url):
    p = urlparse(url)
    return p.netloc == HOST and bool(DRIVER_PATH_RE.match(p.path))


def is_database_url(url):
    """Čokoľvek pod /database/ (zoznamy, jazdci, tímy...): čaká sa na asynchrónne načítanie obsahu."""
    p = urlparse(url)
    return p.netloc == HOST and p.path.startswith("/database/")


def is_database_detail_url(url):
    """/database/*/* (aspoň dva segmenty za database): rozklikávajú sa tu tlačidlá 'View all results'."""
    p = urlparse(url)
    return p.netloc == HOST and bool(DB_DETAIL_RE.match(p.path))


def extract_links(base_url, html):
    found = set()
    for href in HREF_RE.findall(html):
        u = normalize(base_url, href)
        if u and u != base_url:
            found.add(u)
    return sorted(found)    # deterministické poradie


# ---------- stav crawlu (frontier + visited), resumovateľný ----------

class CrawlState:
    """
    frontier  - URL, ktoré čakajú na spracovanie (BFS: berie sa z ľavej strany, DFS: z pravej)
    done      - URL, ktoré sú úplne spracované (stiahnuté + odkazy z nich vytiahnuté)
    failed    - URL, ktoré sa nepodarilo spracovať ani po MAX_RETRIES pokusoch
    discovered = frontier + done + failed (odvodené, nikdy sa nepridá URL dvakrát)
    """

    def __init__(self, strategy="bfs"):
        self.strategy = strategy
        self.frontier = deque()
        self.done = set()
        self.failed = {}
        self.attempts = {}
        self.discovered = set()

    def add(self, url):
        if url in self.discovered:
            return False
        self.discovered.add(url)
        self.frontier.append(url)
        return True

    def take(self):
        return self.frontier.popleft() if self.strategy == "bfs" else self.frontier.pop()

    def give_back(self, url):
        """Prerušené spracovanie (Ctrl+C...): URL sa vráti na miesto, odkiaľ sa vzala."""
        if self.strategy == "bfs":
            self.frontier.appendleft(url)
        else:
            self.frontier.append(url)

    def fail(self, url):
        self.attempts[url] = self.attempts.get(url, 0) + 1
        if self.attempts[url] >= MAX_RETRIES:
            self.failed[url] = self.attempts[url]
        elif self.strategy == "bfs":
            self.frontier.append(url)        # na koniec radu, aby sa nezasekol na jednej stránke
        else:
            self.frontier.appendleft(url)    # na dno zásobníka

    def save(self):
        atomic_write(STATE_FILE, json.dumps({
            "strategy": self.strategy,
            "frontier": list(self.frontier),
            "done": sorted(self.done),
            "failed": self.failed,
            "attempts": self.attempts,
        }, ensure_ascii=False))

    @classmethod
    def load(cls, strategy_override=None, seeds=()):
        st = cls(strategy_override or "bfs")
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.strategy = strategy_override or data.get("strategy", "bfs")
                st.frontier = deque(data["frontier"])
                st.done = set(data["done"])
                st.failed = dict(data.get("failed", {}))
                st.attempts = dict(data.get("attempts", {}))
                st.discovered = set(st.frontier) | st.done | set(st.failed)
                print(f"Pokračujem: hotových {len(st.done)}, vo fronte {len(st.frontier)}, "
                      f"zlyhaných {len(st.failed)} (stratégia {st.strategy.upper()}).")
                added = [u for u in seeds if st.add(u)]    # nové seedy sa pridajú aj pri resume
                if added:
                    print(f"Pridané nové seedy: {len(added)}")
                return st
            except (OSError, ValueError, KeyError):
                print(f"{STATE_FILE} je poškodený, crawl začne odznova (stiahnuté HTML ostáva).")
                st = cls(strategy_override or "bfs")
        for u in seeds:
            st.add(u)
        print(f"Nový crawl, seedov: {len(seeds)} (stratégia {st.strategy.upper()}).")
        return st


# ---------- prehliadač ----------

async def new_page(playwright, headless):
    browser = await playwright.chromium.launch(headless=headless)
    context = await browser.new_context(user_agent=HEADERS["User-Agent"])
    page = await context.new_page()
    await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    await accept_cookies(page)
    return browser, page


async def accept_cookies(page):
    """Cookies lišta (InMobi CMP) prekrýva stránku, tak ju odklikni."""
    try:
        await page.wait_for_selector(".qc-cmp2-persistent-link", timeout=6000)
        await page.click(".qc-cmp2-persistent-link")
        await asyncio.sleep(2)
    except Exception:
        pass


DRIVER_SECTIONS = ("#championship-seasons", "#teams")
ANY_SECTION = ("",)      # prázdny selektor = tlačidlá kdekoľvek na stránke


async def expand_all(page, sections):
    """Click 'View all results' in the given sections (until nothing new loads)."""
    start_url = page.url
    for sec in sections:
        for _ in range(30):
            btn = page.locator(f"{sec} a.button:has-text('View all results')")
            if await btn.count() == 0 or not await btn.first.is_visible():
                break
            before = await page.locator(f"{sec} tbody tr").count()
            await btn.first.click()
            await page.wait_for_timeout(1500)
            if page.url != start_url:
                # tlačidlo bolo obyčajný odkaz a odnavigovalo nás preč, vráť sa, aby obsah patril správnej URL
                await page.go_back(wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
                break
            if await page.locator(f"{sec} tbody tr").count() == before \
                    and await btn.count() > 0:
                break  # nothing new loaded, avoid endless loop


async def goto_database_page(page, url):
    """Otvor stránku a vráť response. Počas navigácie sleduje aj asynchrónne dotiahnutie databázových dát."""
    resp = None
    try:
        async with page.expect_response(
            lambda r: "fetch_database" in r.url, timeout=6000
        ):
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    except PWTimeout:
        if resp is None:
            raise      # nepodarila sa samotná navigácia
        # navigácia prebehla, len stránka žiadne databázové dáta nenačítava, pokračuj
    return resp


async def wait_database_content(page, strict):
    """Počkaj, kým sa vykreslia databázové tabuľky. strict=True (stránka jazdca): chýbajúca hlavička je chyba."""
    if strict:
        await page.wait_for_selector(".database-data__head", timeout=40000)
        for sel in ("#championship-seasons tbody tr", "#teams tbody tr"):
            try:
                await page.wait_for_selector(sel, timeout=7000)
            except PWTimeout:
                pass  # jazdec nemusí mať sezóny alebo tímy
    else:
        try:
            await page.wait_for_selector(".database-data__head, table tbody tr", timeout=5000)
        except PWTimeout:
            pass  # stránka bez tabuliek


def response_usable(resp):
    """False = stránka nemá zmysel spracovať (404, nie HTML). Pri dočasných chybách hodí výnimku (-> retry)."""
    if resp is None:
        return True
    if resp.status in (404, 410):
        return False
    if resp.status >= 400:
        raise RuntimeError(f"HTTP {resp.status}")
    return "html" in (resp.headers.get("content-type") or "html")


# ---------- spracovanie jednej stránky ----------

async def process(page, url):
    """
    Navštívi stránku a vráti odkazy, ktoré sa na nej našli.
      - pod /database/ sa vždy počká na načítanie dát,
      - na /database/*/* sa rozklikne 'View all results',
      - HTML sa ukladá len pri stránkach jazdcov.
    Stránka jazdca, ktorá už je na disku, sa nesťahuje znova, ale odkazy sa z nej aj tak vytiahnu.
    Pri chybe hodí výnimku.
    """
    driver = is_driver_url(url)
    if driver:
        path = html_path(url)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as f:
                return extract_links(url, f.read())

    if is_database_url(url):
        resp = await goto_database_page(page, url)
        if not response_usable(resp):
            return []
        await wait_database_content(page, strict=driver)
    else:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        if not response_usable(resp):
            return []

    if is_database_detail_url(url):
        await expand_all(page, DRIVER_SECTIONS if driver else ANY_SECTION)

    html = await page.content()
    if driver:
        atomic_write(html_path(url), html)
    return extract_links(url, html)


# ---------- crawl ----------

async def crawl(playwright, state, max_pages, headless):
    if not state.frontier:
        print("Fronta je prázdna, crawl je hotový. Na nový crawl použi --reset.")
        return

    browser, page = await new_page(playwright, headless)
    processed = 0
    try:
        while state.frontier and (max_pages is None or processed < max_pages):
            url = state.take()
            ok = False
            try:
                links = await process(page, url)
                ok = True
            except Exception as e:
                print(f"FAIL {url}: {e}")
                state.fail(url)
            except BaseException:
                state.give_back(url)     # Ctrl+C / zrušenie: URL nesmie zmiznúť z fronty
                raise

            if ok:
                state.done.add(url)
                new = [u for u in links if state.add(u)]

                print(f"[{len(state.done)} hotových | {len(state.frontier)} vo fronte] "
                      f"{url} -> {len(new)} nových")

            processed += 1
            if processed % SAVE_STATE_EVERY == 0:
                state.save()
            await asyncio.sleep(SLEEP_TIME)
    finally:
        state.save()                      # uloží sa aj pri Ctrl+C alebo páde
        await browser.close()

    if not state.frontier:
        print(f"Crawl je hotový. Hotových {len(state.done)}, zlyhaných {len(state.failed)}.")
    else:
        print(f"Beh skončil. Hotových {len(state.done)}, vo fronte {len(state.frontier)}, "
              f"zlyhaných {len(state.failed)}. Ďalšie spustenie pokračuje.")


# ---------- spustenie ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Crawler databázy jazdcov (Motor Sport Magazine), BFS/DFS, resumovateľný")
    ap.add_argument("--strategy", choices=["bfs", "dfs"], default=None,
                    help="spôsob prehľadávania (predvolene bfs, pri resume sa použije uložená stratégia)")
    ap.add_argument("--seed", action="append", default=[],
                    help="ďalšia štartovacia URL (dá sa zadať viackrát, pridá sa aj pri pokračovaní)")
    ap.add_argument("--reset", action="store_true",
                    help="zmaž stav crawlu a začni od seedu (stiahnuté HTML ostáva a nesťahuje sa znova)")
    ap.add_argument("--max-pages", type=int, default=None, help="koľko stránok spracovať v tomto behu")
    ap.add_argument("--headless", action="store_true", help="beh bez okna prehliadača")
    return ap.parse_args()


async def run(args):
    max_pages = args.max_pages if args.max_pages is not None else MAX_PAGES_PER_RUN
    seeds = []
    for raw in SEEDS + args.seed:
        u = normalize(BASE_URL, raw)
        if u is None:
            print(f"Seed mimo rozsahu crawlu, preskakujem: {raw}")
        elif u not in seeds:
            seeds.append(u)
    state = CrawlState.load(args.strategy, seeds)
    async with async_playwright() as p:
        await crawl(p, state, max_pages, HEADLESS or args.headless)


def main():
    args = parse_args()
    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"Stav crawlu zmazaný ({STATE_FILE}).")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nPrerušené. Stav je uložený, ďalšie spustenie pokračuje.")


if __name__ == "__main__":
    main()