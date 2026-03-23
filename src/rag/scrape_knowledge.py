"""
scrape_knowledge.py
-------------------
IADC Lexicon full scrape (Parallel & Resumable):
  1. Discover all letter category pages (A-Z, 0-9)
  2. Paginate through each letter
  3. Save all discovered URLs to a JSON state file.
  4. Use ThreadPoolExecutor to visit each term URL and extract definitions.

Uses curl_cffi to bypass bot protection.
"""
import time
import json
from bs4 import BeautifulSoup
from pathlib import Path
import logging
import concurrent.futures
from curl_cffi import requests as cfreq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR  = BASE_DIR / "data" / "knowledge_base" / "raw_text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUT_DIR / "iadc_state.json"
FINAL_FILE = OUT_DIR / "iadc_glossary_full.txt"

# Create a shared session for single-threaded URL discovery
SESSION  = cfreq.Session(impersonate="chrome120")
BASE     = "https://iadclexicon.org"

CATEGORIES = ["0-9"] + list("abcdefghijklmnopqrstuvwxyz")

WIKI_URLS = [
    "https://en.wikipedia.org/wiki/Bottomhole_assembly",
    "https://en.wikipedia.org/wiki/Rate_of_penetration",
    "https://en.wikipedia.org/wiki/Weight_on_bit",
    "https://en.wikipedia.org/wiki/Drill_string",
    "https://en.wikipedia.org/wiki/Drilling_mud",
    "https://en.wikipedia.org/wiki/Blowout_(well_drilling)",
    "https://en.wikipedia.org/wiki/Casing_(borehole)",
    "https://en.wikipedia.org/wiki/Directional_drilling",
]

def get_page(url: str, retries: int = 3, session=None) -> str | None:
    sess = session or SESSION
    for attempt in range(1, retries + 1):
        try:
            r = sess.get(url, timeout=15)
            if r.status_code == 200:
                return r.text
            log.warning(f"[{r.status_code}] {url} (attempt {attempt})")
        except Exception as e:
            log.warning(f"Error {url}: {e} (attempt {attempt})")
        time.sleep(1.5 * attempt)
    return None

def get_all_article_links_from_page(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find(id="content") or soup.find(id="wrap-main-section")
    if not content: return []
    term_links = []
    for article in content.find_all("article"):
        if article.find_parent(id="sidebar-primary"): continue
        for a in article.find_all("a", href=True):
            href = a["href"]
            if href.startswith(BASE) and "/glossary/" not in href and "api.org" not in href:
                term_links.append(href.rstrip("/"))
                break
    return term_links

def get_next_page_url(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    nxt = soup.find("a", class_="next page-numbers")
    if nxt and nxt.get("href"): return nxt["href"]
    return None

def extract_definition(url: str) -> dict | None:
    """Thread-safe extraction using a short-lived local session to avoid cffi thread issues"""
    sess = cfreq.Session(impersonate="chrome120")
    html = get_page(url, session=sess)
    if not html: return None

    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    term_name = h1.get_text(" ", strip=True) if h1 else url.split("/")[-1]

    defn_header = None
    for h3 in soup.find_all("h3"):
        if "Definition" in h3.get_text():
            defn_header = h3
            break

    if defn_header:
        parts = []
        for sibling in defn_header.next_siblings:
            if hasattr(sibling, "has_attr"):
                classes = sibling.get("class", [])
                if "entry-footer" in classes: break
            txt = sibling.get_text("\n", strip=True) if hasattr(sibling, "get_text") else str(sibling).strip()
            if txt: parts.append(txt)
        definition = "\n".join(parts).strip()
    else:
        body = soup.find(class_="entry-content") or soup.find(id="content")
        definition = body.get_text("\n", strip=True) if body else ""

    if not definition: return None
    return {"url": url, "name": term_name, "def": definition}

def scrape_iadc():
    log.info("=== IADC Lexicon Full Crawl ===")
    
    state = {"urls": [], "extracted": {}}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text("utf-8"))
            log.info(f"Loaded existing state: {len(state['urls'])} URLs, {len(state['extracted'])} extracted.")
        except json.JSONDecodeError:
            pass

    all_term_urls = set(state["urls"])
    
    # Phase 1: If we have less than ~5000 URLs, we're probably not done discovering
    # (or if we just want to ensure we have them all)
    # We will resume from where we left off by checking if URLs exist
    # But for simplicity, if we have plenty of URLs already cached, we can skip discovering if it was exhaustive.
    # Instead, let's fast-forward category discovery if we've already done it.
    if len(all_term_urls) < 8000:
        log.info("Discovering URLs...")
        for cat in CATEGORIES:
            page_url = f"{BASE}/glossary/{cat}/"
            page_num = 1
            while page_url:
                log.info(f"  [{cat}] page {page_num} → {page_url}")
                html = get_page(page_url)
                if not html: break

                new_links = get_all_article_links_from_page(html)
                all_term_urls.update(new_links)
                
                # Save state periodically
                state["urls"] = list(all_term_urls)
                STATE_FILE.write_text(json.dumps(state), encoding="utf-8")

                page_url = get_next_page_url(html)
                page_num += 1
                time.sleep(0.5)

    all_term_urls = sorted(all_term_urls)
    log.info(f"\nTotal unique term URLs: {len(all_term_urls)}")

    # Phase 2: extract definitions in parallel
    urls_to_process = [u for u in all_term_urls if u not in state["extracted"]]
    log.info(f"Terms remaining to extract: {len(urls_to_process)}")

    extracted_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(extract_definition, url): url for url in urls_to_process}
        
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                res = future.result()
                if res:
                    state["extracted"][url] = f"TERM: {res['name']}\nURL: {res['url']}\n\n{res['def']}"
                else:
                    state["extracted"][url] = "ERROR: Could not parse"
                
                extracted_count += 1
                if extracted_count % 50 == 0:
                    log.info(f"  Extracted {extracted_count}/{len(urls_to_process)} ...")
                    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
            except Exception as e:
                log.warning(f"Error extracting {url}: {e}")

    # Final save
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    
    # Write output
    valid_records = [v for k, v in state["extracted"].items() if not v.startswith("ERROR")]
    if valid_records:
        FINAL_FILE.write_text("\n\n---\n\n".join(valid_records), encoding="utf-8")
        log.info(f"\nSaved {len(valid_records)} complete terms → {FINAL_FILE.name}")


def scrape_wikipedia():
    log.info("=== Wikipedia Drilling Articles ===")
    for url in WIKI_URLS:
        html = get_page(url)
        if not html: continue
        soup = BeautifulSoup(html, "html.parser")
        content = soup.find(id="mw-content-text")
        if content:
            for noise in content(["script", "style", "table", "div.reflist", "div.navbox"]):
                noise.decompose()
            text = content.get_text("\n", strip=True)
            name = url.split("/")[-1]
            out_path = OUT_DIR / f"wiki_{name}.txt"
            out_path.write_text(f"Source: {url}\n\n{text}", encoding="utf-8")
            log.info(f"  Saved {name}")
        time.sleep(1)

if __name__ == "__main__":
    scrape_iadc()
    scrape_wikipedia()
    log.info("=== Scraping complete ===")
