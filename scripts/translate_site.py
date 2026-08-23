#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse

import requests
import torch
from bs4 import BeautifulSoup, NavigableString
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE = "https://tipitaka.fandom.com"
API = BASE + "/api.php"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGES = ROOT / "pages"
TITLES = DATA / "titles.json"
MANIFEST = DATA / "manifest.json"
PROGRESS = DATA / "progress.json"
FAILURES = DATA / "failures.json"

MODEL_NAME = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-en-hi")
USER_AGENT = "TipitakaHindiFreeCloudTranslator/2.0"
REQUEST_DELAY = float(os.getenv("SOURCE_REQUEST_DELAY", "0.15"))
MAX_SOURCE_WORDS = int(os.getenv("MAX_SOURCE_WORDS", "110"))
MODEL_BATCH = int(os.getenv("MODEL_BATCH", "8"))

VRI_REPLACEMENTS = [
    (r"त्रिपिटक", "तिपिटक"),
    (r"टिपिटक", "तिपिटक"),
    (r"पाली", "पालि"),
    (r"दीर्घ निकाय", "दीघ निकाय"),
    (r"मध्य(?:म)? निकाय", "मज्झिम निकाय"),
    (r"संयुक्त निकाय", "संयुत्त निकाय"),
    (r"अंगुत्तर निकाय", "अङ्गुत्तर निकाय"),
    (r"\bसूत्र\b", "सुत्त"),
    (r"सूत्त", "सुत्त"),
    (r"सुत्त पिटक", "सुत्तपिटक"),
    (r"निब्बान", "निर्वाण"),
    (r"निर्वाना", "निर्वाण"),
    (r"अरहन्त", "अर्हत"),
    (r"अरहंत", "अर्हत"),
    (r"भिक्खु", "भिक्षु"),
    (r"समभाव", "समता"),
    (r"अस्थायित्व", "अनिच्च (अनित्यता)"),
    (r"अनित्यता \(अनिच्च\)", "अनिच्च (अनित्यता)"),
    (r"अनात्मा", "अनत्त"),
    (r"लालसा", "तृष्णा"),
]

PALI_MARKERS = {
    "bhikkhave", "bhagavā", "evaṃ", "suttaṃ", "dhammaṃ", "saṅghaṃ",
    "tathāgato", "arahaṃ", "sammāsambuddho", "dukkhaṃ", "aniccaṃ",
    "anattā", "vedanā", "taṇhā", "nibbānaṃ",
}

REMOVE_SELECTORS = [
    "script", "style", "noscript", "iframe", "form",
    ".mw-editsection", ".noprint", ".mw-empty-elt",
    ".reference-backlink"
]

PRIORITY_TITLES = [
    "Main Page",
    "Tipitaka",
    "Vinaya Pitaka",
    "Sutta Pitaka",
    "Abhidhamma Pitaka",
    "Digha Nikaya",
    "Majjhima Nikaya",
    "Samyutta Nikaya",
    "Anguttara Nikaya",
    "Khuddaka Nikaya",
]

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def normalize_hi(text: str) -> str:
    out = text
    for pattern, repl in VRI_REPLACEMENTS:
        out = re.sub(pattern, repl, out)
    return out

def dev_ratio(text: str) -> float:
    letters = sum(ch.isalpha() for ch in text)
    if not letters:
        return 0.0
    dev = sum("\u0900" <= ch <= "\u097F" for ch in text)
    return dev / letters

def looks_pali(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 60:
        return False
    words = set(re.findall(r"[\wāīūṃṅñṭḍṇḷṛ]+", t, flags=re.UNICODE))
    marker_hits = len(words & PALI_MARKERS)
    diacritics = sum(t.count(ch) for ch in "āīūṃṅñṭḍṇḷ")
    return marker_hits >= 2 or diacritics >= 6

def should_translate(text: str, parent_name: str | None, page_title: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if parent_name in {"script", "style", "code", "pre", "textarea", "option"}:
        return False
    if not any(ch.isalpha() for ch in t):
        return False
    if dev_ratio(t) >= 0.55:
        return False
    if re.fullmatch(r"https?://\S+", t):
        return False
    if re.search(r"pali|pāḷi|devanagri version|roman version", page_title, flags=re.I):
        return False
    if looks_pali(t):
        return False
    return True

def api_get(session: requests.Session, params: dict, retries: int = 5) -> dict:
    p = dict(params)
    p["format"] = "json"
    last = None
    for attempt in range(retries):
        try:
            r = session.get(API, params=p, timeout=90)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"].get("info", str(data["error"])))
            return data
        except Exception as e:
            last = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Fandom API failed: {last}")

def discover_titles(session: requests.Session) -> list[dict]:
    out = []
    cont = {}
    while True:
        data = api_get(session, {
            "action": "query",
            "list": "allpages",
            "apnamespace": 0,
            "aplimit": "max",
            **cont,
        })
        for p in data.get("query", {}).get("allpages", []):
            out.append({"pageid": p.get("pageid"), "title": p["title"]})
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(REQUEST_DELAY)
    return out

def fetch_page(session: requests.Session, title: str) -> tuple[str, str, str]:
    data = api_get(session, {
        "action": "parse",
        "page": title,
        "prop": "text|displaytitle",
        "disableeditsection": 1,
        "redirects": 1,
    })
    p = data["parse"]
    canonical = p.get("title") or title
    display = p.get("displaytitle") or canonical
    return canonical, display, p["text"]["*"]

def clean_html(raw: str) -> BeautifulSoup:
    soup = BeautifulSoup(raw, "html.parser")
    for selector in REMOVE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            k = attr.lower()
            if k.startswith("on") or k in {"srcset", "style", "data-src", "data-image-key", "data-image-name"}:
                del tag.attrs[attr]
    return soup

class FreeTranslator:
    def __init__(self):
        print(f"Loading free translation model: {MODEL_NAME}")
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.model.eval()

    def split_text(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?।])\s+", text)
        out, current, words = [], [], 0
        for sentence in sentences:
            sw = sentence.split()
            if len(sw) > MAX_SOURCE_WORDS:
                if current:
                    out.append(" ".join(current))
                    current, words = [], 0
                for i in range(0, len(sw), MAX_SOURCE_WORDS):
                    out.append(" ".join(sw[i:i + MAX_SOURCE_WORDS]))
                continue
            if current and words + len(sw) > MAX_SOURCE_WORDS:
                out.append(" ".join(current))
                current, words = [], 0
            current.append(sentence)
            words += len(sw)
        if current:
            out.append(" ".join(current))
        return out

    def translate_chunks(self, chunks: list[str]) -> list[str]:
        results = []
        for start in range(0, len(chunks), MODEL_BATCH):
            batch = chunks[start:start + MODEL_BATCH]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.inference_mode():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=512,
                    num_beams=1,
                    do_sample=False,
                )
            results.extend(
                normalize_hi(x)
                for x in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return results

    def translate_text(self, text: str) -> str:
        chunks = self.split_text(text)
        if not chunks:
            return text
        return " ".join(self.translate_chunks(chunks))

def extract_nodes(soup: BeautifulSoup, page_title: str):
    nodes = []
    for node in soup.find_all(string=True):
        if not isinstance(node, NavigableString):
            continue
        parent = node.parent
        if parent is None:
            continue
        if should_translate(str(node), parent.name, page_title):
            nodes.append(node)
    return nodes

def apply_translation(soup: BeautifulSoup, translator: FreeTranslator, page_title: str) -> tuple[int, int]:
    nodes = extract_nodes(soup, page_title)
    total = len(nodes)
    done = 0
    for node in nodes:
        raw = str(node)
        core = raw.strip()
        lead = raw[:len(raw) - len(raw.lstrip())]
        trail = raw[len(raw.rstrip()):]
        translated = translator.translate_text(core)
        node.replace_with(lead + translated + trail)
        done += 1
    return done, total

def rewrite_links_and_images(soup: BeautifulSoup):
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        try:
            u = urlparse(urljoin(BASE, href))
        except Exception:
            continue
        if u.netloc == "tipitaka.fandom.com" and u.path.startswith("/wiki/"):
            title = unquote(u.path[len("/wiki/"):]).replace("_", " ")
            a["href"] = "../index.html?title=" + quote(title, safe="")
            if u.fragment:
                a["href"] += "#" + u.fragment
        elif href.startswith("/"):
            a["href"] = urljoin(BASE, href)
    for img in soup.find_all("img", src=True):
        src = img.get("src", "")
        if src.startswith("/"):
            img["src"] = urljoin(BASE, src)
        img["loading"] = "lazy"

PAGE_CSS = """
:root{--paper:#fffdf7;--ink:#292018;--muted:#74685d;--deep:#63350f;--line:#e5d4bd;--soft:#f8efe3}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Noto Sans Devanagari","Nirmala UI","Mangal",system-ui,sans-serif;line-height:1.78}
.top{position:sticky;top:0;z-index:20;background:#fffaf2ef;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:10px 14px}.top a{color:var(--deep);text-decoration:none;font-weight:700;margin-right:16px}
main{max-width:980px;margin:auto;padding:26px 18px 70px;background:#fff}.crumb{color:var(--muted);font-size:.9rem}.title{font-size:clamp(2rem,5vw,3.5rem);line-height:1.25;color:var(--deep);margin:.35em 0}
article h1,article h2,article h3,article h4{color:#713e10;line-height:1.4}article h2{border-bottom:1px solid var(--line);padding-bottom:7px}article p,article li{font-size:1.05rem}
article table{border-collapse:collapse;max-width:100%;display:block;overflow:auto}article th,article td{border:1px solid #d6c5af;padding:7px 9px}article img{max-width:100%;height:auto}
.source{margin-top:36px;padding:15px;background:var(--soft);border-radius:10px;color:var(--muted);font-size:.9rem}.source a{color:#74400e}
"""

def make_page(source_title: str, hindi_title: str, article_html: str, source_url: str, blocks: int) -> str:
    return f"""<!doctype html>
<html lang="hi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(hindi_title)} | तिपिटक हिंदी</title>
<meta name="description" content="Wikipitaka स्रोत का पूर्ण VRI-शैली हिंदी रूपांतरण">
<style>{PAGE_CSS}</style>
</head>
<body>
<nav class="top"><a href="../index.html">☸ तिपिटक हिंदी</a><a href="../index.html">मुख्य सूची</a></nav>
<main>
<div class="crumb">Wikipitaka → पूर्ण हिंदी पृष्ठ</div>
<h1 class="title">{html.escape(hindi_title)}</h1>
<article>{article_html}</article>
<div class="source"><strong>मूल स्रोत:</strong> <a href="{html.escape(source_url)}" target="_blank" rel="noopener">{html.escape(source_title)}</a><br>
यह स्रोत-पृष्ठ का स्वतंत्र पूर्ण हिंदी रूपांतरण है, VRI-शैली शब्दावली के अनुरूप। यह VRI द्वारा प्रमाणित/अधिकृत अनुवाद होने का दावा नहीं करता।<br>
अनूदित text blocks: {blocks}. अनुवाद मुक्त open-source मॉडल {html.escape(MODEL_NAME)} से GitHub Actions पर किया गया है।</div>
</main>
</body>
</html>"""

def translate_page(session: requests.Session, translator: FreeTranslator, page: dict):
    title = page["title"]
    pageid = page.get("pageid") or abs(hash(title))
    canonical, display, raw = fetch_page(session, title)
    soup = clean_html(raw)
    done, total = apply_translation(soup, translator, canonical)
    rewrite_links_and_images(soup)

    display_text = BeautifulSoup(display, "html.parser").get_text(" ", strip=True)
    if should_translate(display_text, None, canonical):
        hindi_title = translator.translate_text(display_text)
    else:
        hindi_title = display_text
    hindi_title = normalize_hi(hindi_title)

    source_url = BASE + "/wiki/" + quote(canonical.replace(" ", "_"), safe="():,._-")
    filename = f"{pageid}.html"
    page_html = make_page(canonical, hindi_title, str(soup), source_url, done)
    (PAGES / filename).write_text(page_html, encoding="utf-8")
    return {
        "title": title,
        "canonical_title": canonical,
        "display_title": hindi_title,
        "source_display_title": display_text,
        "file": f"pages/{filename}",
        "source": source_url,
        "translated_blocks": done,
        "total_blocks": total,
        "updated_at": utcnow(),
        "model": MODEL_NAME,
        "engine": "free-open-source",
    }

def priority_key(page: dict, order: dict[str, int], failures: dict) -> tuple:
    title = page["title"]
    if title in PRIORITY_TITLES:
        p = (0, PRIORITY_TITLES.index(title))
    elif re.search(r"\b(?:DN|MN|SN|AN)\s*\d|Sutta\b|Nikaya\b|Pitaka\b", title, flags=re.I):
        p = (1, order[title])
    else:
        p = (2, order[title])
    attempts = int(failures.get(title, {}).get("attempts", 0))
    return (p[0], attempts, p[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "5")))
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    PAGES.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    titles = load_json(TITLES, [])
    if not titles:
        print("Discovering Wikipitaka namespace-0 pages...")
        titles = discover_titles(session)
        save_json(TITLES, titles)

    manifest = load_json(MANIFEST, {})
    failures = load_json(FAILURES, {})
    completed = set(manifest)

    order = {p["title"]: i for i, p in enumerate(titles)}
    candidates = [p for p in titles if p["title"] not in completed]
    candidates.sort(key=lambda p: priority_key(p, order, failures))
    selected = candidates[:max(1, args.batch_size)]

    print(f"Total={len(titles)} Completed={len(completed)} Selected={len(selected)} Model={MODEL_NAME}")

    translator = FreeTranslator() if selected else None
    batch_succeeded = 0

    for n, page in enumerate(selected, 1):
        title = page["title"]
        print(f"[{n}/{len(selected)}] {title}")
        try:
            manifest[title] = translate_page(session, translator, page)
            failures.pop(title, None)
            batch_succeeded += 1
            save_json(MANIFEST, manifest)
            save_json(FAILURES, failures)
        except Exception as e:
            rec = failures.get(title, {"attempts": 0})
            rec["attempts"] = int(rec.get("attempts", 0)) + 1
            rec["last_error"] = str(e)[:1000]
            rec["updated_at"] = utcnow()
            failures[title] = rec
            save_json(FAILURES, failures)
            print(f"ERROR: {title}: {e}")
        time.sleep(REQUEST_DELAY)

    completed_count = len(manifest)
    progress = {
        "status": "complete" if completed_count >= len(titles) else "running",
        "total": len(titles),
        "completed": completed_count,
        "remaining": max(0, len(titles) - completed_count),
        "failed_pending": len([k for k in failures if k not in manifest]),
        "percent": round((completed_count / len(titles) * 100), 3) if titles else 0,
        "updated_at": utcnow(),
        "model": MODEL_NAME,
        "engine": "free-open-source",
        "batch_size": args.batch_size,
        "last_batch_selected": len(selected),
        "last_batch_succeeded": batch_succeeded,
    }
    save_json(PROGRESS, progress)
    print(json.dumps(progress, ensure_ascii=False))

    if selected and batch_succeeded == 0:
        raise SystemExit("No pages translated successfully in this batch.")

if __name__ == "__main__":
    main()
