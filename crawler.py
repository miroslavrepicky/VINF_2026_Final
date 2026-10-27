"""
Crawler for the Motor Sport Magazine driver database (Playwright, resumable, BFS/DFS).

    python3 crawler.py                       (start / resume the crawl)
    python3 crawler.py --strategy dfs        (depth-first instead of breadth-first)
    python3 crawler.py --seed <url>          (add another start URL)
    python3 crawler.py --max-pages 100       (process at most 100 pages in this run)
    python3 crawler.py --reset               (drop the crawl state, start from the seeds)

Only driver pages (/database/drivers/<slug>/) are saved to disk as HTML; every other page
under the host is visited only to discover further links. The crawl state (frontier, done,
failed) is stored in crawl_state.json so an interrupted run continues where it stopped.
"""
import argparse
import asyncio
import json
import os
import re
from collections import deque
from urllib.parse import urljoin, urlparse, parse_qs
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ---------- configuration ----------
SITE = "https://www.motorsportmagazine.com"
BASE_URL = SITE + "/database/drivers/"
SEEDS = [BASE_URL]                               # start pages (more can be added with --seed)
HOST = "www.motorsportmagazine.com"
ALLOW_EXTERNAL = False   # False = crawl the whole HOST domain (all paths); True = also follow links to other domains
KEEP_QUERY_PARAMS = ("page",)   # only these query parameters are kept (the rest would create endless URL variants)
SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf", ".zip", ".mp4", ".mp3",
                   ".css", ".js", ".json", ".xml", ".woff", ".woff2", ".ttf", ".avif", ".rss")
STATE_FILE = "crawl_state.json"                  # frontier + done + failed (for resuming)
SAVE_DIR = "data/html"
HEADERS = {
    "User-Agent": "VINF-Project-Crawler/1.0 (student project at STU FIIT Bratislava, course VINF; contact: xrepickym@stuba.sk)"
}
SLEEP_TIME = 1.5          # politeness delay between two page visits (seconds)
HEADLESS = False          # --headless switch; driver pages were downloaded with a visible window, so that is the default
MAX_PAGES_PER_RUN = None  # pages per run (None = no limit), overridden by --max-pages
MAX_RETRIES = 3           # number of failed attempts after which a URL is given up on
SAVE_STATE_EVERY = 10     # how often the state is saved (it is always saved on exit as well)
NAV_TIMEOUT = 5000        # navigation timeout in milliseconds

DRIVER_PATH_RE = re.compile(r"^/database/drivers/[a-z0-9\-]+/$")   # a single driver page
DB_DETAIL_RE = re.compile(r"^/database/[^/]+/[^/]+")                # /database/<type>/<something>
HREF_RE = re.compile(r'<a\s[^>]*?href=["\']([^"\'#]+)', re.I | re.S)


# ---------- files ----------

def atomic_write(path, text):
    """Write to a temporary file and rename it, so a crash mid-write cannot leave a corrupted file."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def html_path(url):
    """Map a URL to the local file where its HTML is stored (path segments joined by '_')."""
    filename = urlparse(url).path.strip("/").replace("/", "_") or "index"
    return f"{SAVE_DIR}/{filename}.html"


# ---------- URLs: normalization and crawl scope ----------

def normalize(base, href):
    """Return the canonical URL if it should be crawled, otherwise None."""
    url = urljoin(base, href.strip().replace("&amp;", "&"))
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        return None                                  # mailto:, javascript:, tel: ...
    if not ALLOW_EXTERNAL and p.netloc != HOST:
        return None
    if p.path.lower().endswith(SKIP_EXTENSIONS):
        return None                                  # images, pdf, css... pointless to open in a browser
    path = p.path or "/"
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"                                  # /x and /x/ are the same page (files with an extension are left alone)
    if p.netloc == HOST and DRIVER_PATH_RE.match(path):
        return f"{SITE}{path}"                       # driver page: no query string
    qs = parse_qs(p.query)
    kept = []
    for k in KEEP_QUERY_PARAMS:
        v = qs.get(k, [None])[0]
        if v and not (k == "page" and v == "1"):     # page=1 is the same as no parameter
            kept.append(f"{k}={v}")
    return f"{p.scheme}://{p.netloc}{path}" + ("?" + "&".join(kept) if kept else "")


def is_driver_url(url):
    """True for a single driver page (/database/drivers/<slug>/)."""
    p = urlparse(url)
    return p.netloc == HOST and bool(DRIVER_PATH_RE.match(p.path))


def is_database_url(url):
    """Anything under /database/ (lists, drivers, teams...): its content is loaded asynchronously, so we wait for it."""
    p = urlparse(url)
    return p.netloc == HOST and p.path.startswith("/database/")


def is_database_detail_url(url):
    """/database/*/* (at least two segments after 'database'): the 'View all results' buttons are clicked here."""
    p = urlparse(url)
    return p.netloc == HOST and bool(DB_DETAIL_RE.match(p.path))


def extract_links(base_url, html):
    """Return all crawlable links found in the HTML (normalized, deduplicated, sorted)."""
    found = set()
    for href in HREF_RE.findall(html):
        u = normalize(base_url, href)
        if u and u != base_url:
            found.add(u)
    return sorted(found)    # deterministic order


# ---------- crawl state (frontier + visited), resumable ----------

class CrawlState:
    """
    frontier  - URLs waiting to be processed (BFS: taken from the left side, DFS: from the right side)
    done      - URLs that are fully processed (downloaded + links extracted)
    failed    - URLs that could not be processed even after MAX_RETRIES attempts
    discovered = frontier + done + failed (derived; a URL is never added twice)
    """

    def __init__(self, strategy="bfs"):
        self.strategy = strategy
        self.frontier = deque()
        self.done = set()
        self.failed = {}
        self.attempts = {}
        self.discovered = set()

    def add(self, url):
        """Add a URL to the frontier unless it was already discovered. Returns True if it was new."""
        if url in self.discovered:
            return False
        self.discovered.add(url)
        self.frontier.append(url)
        return True

    def take(self):
        """Take the next URL from the frontier (queue for BFS, stack for DFS)."""
        return self.frontier.popleft() if self.strategy == "bfs" else self.frontier.pop()

    def give_back(self, url):
        """Interrupted processing (Ctrl+C...): the URL is put back where it was taken from."""
        if self.strategy == "bfs":
            self.frontier.appendleft(url)
        else:
            self.frontier.append(url)

    def fail(self, url):
        """Record a failed attempt; re-queue the URL or give up on it after MAX_RETRIES attempts."""
        self.attempts[url] = self.attempts.get(url, 0) + 1
        if self.attempts[url] >= MAX_RETRIES:
            self.failed[url] = self.attempts[url]
        elif self.strategy == "bfs":
            self.frontier.append(url)        # to the end of the queue so we do not get stuck on one page
        else:
            self.frontier.appendleft(url)    # to the bottom of the stack

    def save(self):
        """Persist the whole crawl state to STATE_FILE (atomically)."""
        atomic_write(STATE_FILE, json.dumps({
            "strategy": self.strategy,
            "frontier": list(self.frontier),
            "done": sorted(self.done),
            "failed": self.failed,
            "attempts": self.attempts,
        }, ensure_ascii=False))

    @classmethod
    def load(cls, strategy_override=None, seeds=()):
        """Load the saved state (resume) or start a new one; new seeds are added in both cases."""
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
                print(f"Resuming: {len(st.done)} done, {len(st.frontier)} in frontier, "
                      f"{len(st.failed)} failed ({st.strategy.upper()} strategy).")
                added = [u for u in seeds if st.add(u)]    # new seeds are added on resume as well
                if added:
                    print(f"New seeds added: {len(added)}")
                return st
            except (OSError, ValueError, KeyError):
                print(f"{STATE_FILE} is corrupted, the crawl starts from scratch (downloaded HTML is kept).")
                st = cls(strategy_override or "bfs")
        for u in seeds:
            st.add(u)
        print(f"New crawl, seeds: {len(seeds)} ({st.strategy.upper()} strategy).")
        return st


# ---------- browser ----------

async def new_page(playwright, headless):
    """Launch Chromium, open the base page and dismiss the cookie banner. Returns (browser, page)."""
    browser = await playwright.chromium.launch(headless=headless)
    context = await browser.new_context(user_agent=HEADERS["User-Agent"])
    page = await context.new_page()
    await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    await accept_cookies(page, headless)
    return browser, page


COOKIE_BANNER = ".qc-cmp2-container"     # InMobi CMP banner container


async def accept_cookies(page, headless):
    """
    The cookie banner (InMobi CMP) covers the page. Automatic clicking does not work on it,
    so the user has to accept it manually in the browser window; the crawler just waits
    (with no time limit) until the banner disappears.
    """
    try:
        await page.wait_for_selector(COOKIE_BANNER, state="visible", timeout=6000)
    except PWTimeout:
        return      # no banner (already accepted or not shown)

    if headless:
        print("WARNING: the cookie banner is shown, but the browser is headless, so it cannot be "
              "accepted manually. Run without --headless.")
        return

    print(">>> Cookie banner detected. Accept it in the browser window; the crawl continues afterwards.")
    await page.wait_for_selector(COOKIE_BANNER, state="hidden", timeout=0)   # timeout=0 = wait forever
    print(">>> Cookie banner accepted, continuing.")
    await asyncio.sleep(2)


DRIVER_SECTIONS = ("#championship-seasons", "#teams")   # sections of a driver page with 'View all results' buttons
ANY_SECTION = ("",)      # empty selector = buttons anywhere on the page


async def expand_all(page, sections):
    """Click 'View all results' in the given sections (until nothing new loads)."""
    start_url = page.url
    for sec in sections:
        for _ in range(30):                          # hard cap on the number of clicks per section
            btn = page.locator(f"{sec} a.button:has-text('View all results')")
            if await btn.count() == 0 or not await btn.first.is_visible():
                break
            before = await page.locator(f"{sec} tbody tr").count()
            await btn.first.click()
            await page.wait_for_timeout(1500)
            if page.url != start_url:
                # the button was a plain link and navigated away; go back so the content belongs to the right URL
                await page.go_back(wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
                break
            if await page.locator(f"{sec} tbody tr").count() == before \
                    and await btn.count() > 0:
                break  # nothing new loaded, avoid an endless loop


async def goto_database_page(page, url):
    """Open the page and return the response. Also watches for the asynchronous loading of the database data."""
    resp = None
    try:
        async with page.expect_response(
            lambda r: "fetch_database" in r.url, timeout=6000
        ):
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    except PWTimeout:
        if resp is None:
            raise      # the navigation itself failed
        # the navigation succeeded, the page just does not load any database data; carry on
    return resp


async def wait_database_content(page, strict):
    """Wait until the database tables are rendered. strict=True (driver page): a missing header is an error."""
    if strict:
        await page.wait_for_selector(".database-data__head", timeout=40000)
        for sel in ("#championship-seasons tbody tr", "#teams tbody tr"):
            try:
                await page.wait_for_selector(sel, timeout=7000)
            except PWTimeout:
                pass  # a driver may have no seasons or teams
    else:
        try:
            await page.wait_for_selector(".database-data__head, table tbody tr", timeout=5000)
        except PWTimeout:
            pass  # page without tables


def response_usable(resp):
    """False = the page is not worth processing (404, not HTML). Temporary errors raise (-> retry)."""
    if resp is None:
        return True
    if resp.status in (404, 410):
        return False
    if resp.status >= 400:
        raise RuntimeError(f"HTTP {resp.status}")
    return "html" in (resp.headers.get("content-type") or "html")


# ---------- processing a single page ----------

async def process(page, url):
    """
    Visit a page and return the links found on it.
      - under /database/ it always waits for the data to load,
      - on /database/*/* it clicks 'View all results',
      - HTML is saved only for driver pages.
    A driver page that is already on disk is not downloaded again, but its links are still extracted.
    Raises an exception on failure.
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
    """Main crawl loop: take a URL, process it, enqueue new links, save the state periodically."""
    if not state.frontier:
        print("The frontier is empty, the crawl is finished. Use --reset for a new crawl.")
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
                state.give_back(url)     # Ctrl+C / cancellation: the URL must not vanish from the frontier
                raise

            if ok:
                state.done.add(url)
                new = [u for u in links if state.add(u)]

                print(f"[{len(state.done)} done | {len(state.frontier)} in frontier] "
                      f"{url} -> {len(new)} new")

            processed += 1
            if processed % SAVE_STATE_EVERY == 0:
                state.save()
            await asyncio.sleep(SLEEP_TIME)
    finally:
        state.save()                      # saved on Ctrl+C or a crash as well
        await browser.close()

    if not state.frontier:
        print(f"Crawl finished. {len(state.done)} done, {len(state.failed)} failed.")
    else:
        print(f"Run finished. {len(state.done)} done, {len(state.frontier)} in frontier, "
              f"{len(state.failed)} failed. The next run continues from here.")


# ---------- entry point ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Driver database crawler (Motor Sport Magazine), BFS/DFS, resumable")
    ap.add_argument("--strategy", choices=["bfs", "dfs"], default=None,
                    help="crawl strategy (BFS by default; on resume the saved strategy is used)")
    ap.add_argument("--seed", action="append", default=[],
                    help="additional start URL (can be given multiple times, also added when resuming)")
    ap.add_argument("--reset", action="store_true",
                    help="delete the crawl state and start from the seeds (downloaded HTML is kept and not downloaded again)")
    ap.add_argument("--max-pages", type=int, default=None, help="how many pages to process in this run")
    ap.add_argument("--headless", action="store_true", help="run without a browser window")
    return ap.parse_args()


async def run(args):
    """Build the seed list, load the crawl state and run the crawl."""
    max_pages = args.max_pages if args.max_pages is not None else MAX_PAGES_PER_RUN
    seeds = []
    for raw in SEEDS + args.seed:
        u = normalize(BASE_URL, raw)
        if u is None:
            print(f"Seed outside the crawl scope, skipping: {raw}")
        elif u not in seeds:
            seeds.append(u)
    state = CrawlState.load(args.strategy, seeds)
    async with async_playwright() as p:
        await crawl(p, state, max_pages, HEADLESS or args.headless)


def main():
    args = parse_args()
    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"Crawl state deleted ({STATE_FILE}).")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted. The state is saved, the next run continues from here.")


if __name__ == "__main__":
    main()