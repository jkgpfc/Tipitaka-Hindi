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
from typing import Iterable
from urllib.parse import quote, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString
from openai import OpenAI

BASE = "https://tipitaka.fandom.com"
API = BASE + "/api.php"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGES = ROOT / "pages"
TITLES = DATA / "titles.json"
MANIFEST = DATA / "manifest.json"
PROGRESS = DATA / "progress.json"
FAILURES = DATA / "failures.json"

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
USER_AGENT = "TipitakaHindiCloudTranslator/1.0 (GitHub Pages translation project)"
MAX_BATCH_CHARS = int(os.getenv("MAX_BATCH_CHARS", "12000"))
REQUEST_DELAY = float(os.getenv("SOURCE_REQUEST_DELAY", "0.15"))

VRI_REPLACEMENTS = [
    (r"त्रिपिटक", "तिपिटक"),
    (r"पाली", "पालि"),
    (r"दीर्घ निकाय", "दीघ निकाय"),
    (r"मध्य(?:म)? निकाय", "मज्झिम निकाय"),
    (r"संयुक्त निकाय", "संयुत्त निकाय"),
    (r"अंगुत्तर निकाय", "अङ्गुत्तर निकाय"),
    (r"\bसूत्र\b", "सुत्त"),
    (r"निब्बान", "निर्वाण"),
    (r"समभाव", "समता"),
    (r"अस्थायित्व", "अनिच्च (अनित्यता)"),
    (r"अनित्यता \(अनिच्च\)", "अनिच्च (अनित्यता)"),
    (r"अनात्मा", "अनत्त"),
    (r"लालसा", "तृष्णा"),
]

SYSTEM = """You are translating Theravada Buddhist canonical and study material from English into Hindi.

Translate EVERY supplied segment fully. Do not summarize, shorten, omit, add commentary, or combine segments.
Use sober Hindi in the publication style associated with Vipassana Research Institute terminology, while making no claim of VRI endorsement.

Mandatory terminology preferences:
Tipitaka/Tipiṭaka = तिपिटक
Pali/Pāḷi = पालि
Dhamma = धम्म
Sutta = सुत्त
Vinaya Piṭaka = विनय पिटक
Sutta Piṭaka = सुत्तपिटक
Abhidhamma Piṭaka = अभिधम्म पिटक
Dīgha Nikāya = दीघ निकाय
Majjhima Nikāya = मज्झिम निकाय
Saṃyutta Nikāya = संयुत्त निकाय
Aṅguttara Nikāya = अङ्गुत्तर निकाय
Khuddaka Nikāya = खुद्दक निकाय
sīla = शील
samādhi = समाधि
paññā = प्रज्ञा (पञ्ञा)
anicca = अनिच्च (अनित्यता)
dukkha = दुःख
anattā = अनत्त
taṇhā = तृष्णा
vedanā = वेदना (संवेदना)
upekkhā = उपेक्खा (समता)
sati = सति
sampajañña = सम्पजञ्ञ
paṭiccasamuppāda = प्रतीत्य-समुत्पाद
Nibbāna = निर्वाण
bhikkhu = भिक्षु
arahant = अर्हत
Sammā Sambuddha = सम्यक सम्बुद्ध

Preserve Pali proper names and canonical identifiers such as DN, MN, SN, AN, KN.
Do not translate URLs, reference codes, or strings that are already predominantly Devanagari.
Return ONLY valid JSON in exactly this shape:
{"items":[{"id":0,"hi":"..."},{"id":1,"hi":"..."}]}
The id values and count must exactly match the input.
"""

REMOVE_SELECTORS = [
    "script", "style", "noscript", "iframe", "form",
    ".mw-editsection", ".noprint", ".mw-empty-elt",
    ".reference-backlink"
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

def should_translate(text: str, parent_name: str | None) -> bool:
    t = text.strip()
    if not t:
        return False
    if parent_name in {"script", "style", "code", "pre", "textarea", "option"}:
        return False
    if not any(ch.isalpha() for ch in t):
        return False
    if dev_ratio(t) >= 0.65:
        return False
    if re.fullmatch(r"https?://\S+", t):
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

def extract_nodes(soup: BeautifulSoup):
    nodes = []
    for node in soup.find_all(string=True):
        if not isinstance(node, NavigableString):
            continue
        parent = node.parent
        if parent is None:
            continue
        if should_translate(str(node), parent.name):
            nodes.append(node)
    return nodes

def make_batches(nodes) -> Iterable[list[tuple[int, str]]]:
    batch = []
    chars = 0
    for idx, node in enumerate(nodes):
        text = str(node).strip()
        if not text:
            continue
        if batch and chars + len(text) > MAX_BATCH_CHARS:
            yield batch
            batch = []
            chars = 0
        batch.append((idx, text))
        chars += len(text)
    if batch:
        yield batch

def parse_json_output(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def translate_one(client: OpenAI, text: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=SYSTEM + "\n\nINPUT:\n" + json.dumps({"items":[{"id":0,"text":text}]}, ensure_ascii=False),
    )
    try:
        obj = parse_json_output(response.output_text)
        return normalize_hi(obj["items"][0]["hi"])
    except Exception:
        return normalize_hi(response.output_text.strip())

def translate_batch(client: OpenAI, batch: list[tuple[int, str]]) -> dict[int, str]:
    payload = {"items": [{"id": idx, "text": text} for idx, text in batch]}
    response = client.responses.create(
        model=MODEL,
        input=SYSTEM + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False),
    )
    try:
        obj = parse_json_output(response.output_text)
        items = obj["items"]
        mapping = {int(x["id"]): normalize_hi(str(x["hi"])) for x in items}
        expected = {idx for idx, _ in batch}
        if set(mapping) != expected:
            raise ValueError("Returned ids do not match input")
        return mapping
    except Exception:
        return {idx: translate_one(client, text) for idx, text in batch}

def apply_translation(soup: BeautifulSoup, client: OpenAI) -> tuple[int, int]:
    nodes = extract_nodes(soup)
    total = len(nodes)
    done = 0
    for batch in make_batches(nodes):
        mapping = translate_batch(client, batch)
        for idx, _ in batch:
            node = nodes[idx]
            raw = str(node)
            lead = raw[:len(raw)-len(raw.lstrip())]
            trail = raw[len(raw.rstrip()):]
            node.replace_with(lead + mapping[idx] + trail)
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

def make_page(title: str, display: str, article_html: str, source_url: str, blocks: int) -> str:
    return f"""<!doctype html>
<html lang="hi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} | तिपिटक हिंदी</title>
<meta name="description" content="Wikipitaka स्रोत का पूर्ण VRI-शैली हिंदी रूपांतरण">
<style>{PAGE_CSS}</style>
</head>
<body>
<nav class="top"><a href="../index.html">☸ तिपिटक हिंदी</a><a href="../index.html">मुख्य सूची</a></nav>
<main>
<div class="crumb">Wikipitaka → पूर्ण हिंदी पृष्ठ</div>
<h1 class="title">{display}</h1>
<article>{article_html}</article>
<div class="source"><strong>मूल स्रोत:</strong> <a href="{html.escape(source_url)}" target="_blank" rel="noopener">{html.escape(title)}</a><br>
यह स्रोत-पृष्ठ का स्वतंत्र पूर्ण हिंदी रूपांतरण है, VRI-शैली शब्दावली के अनुरूप। यह VRI द्वारा प्रमाणित/अधिकृत अनुवाद होने का दावा नहीं करता।<br>
अनूदित text blocks: {blocks}. स्रोत पृष्ठ पर लागू लाइसेंस/attribution शर्तों का सम्मान किया गया है।</div>
</main>
</body>
</html>"""

def translate_page(session, client, page: dict):
    title = page["title"]
    pageid = page.get("pageid") or abs(hash(title))
    canonical, display, raw = fetch_page(session, title)
    soup = clean_html(raw)
    done, total = apply_translation(soup, client)
    rewrite_links_and_images(soup)
    source_url = BASE + "/wiki/" + quote(canonical.replace(" ", "_"), safe="():,._-")
    filename = f"{pageid}.html"
    page_html = make_page(canonical, display, str(soup), source_url, done)
    (PAGES / filename).write_text(page_html, encoding="utf-8")
    return {
        "title": title,
        "canonical_title": canonical,
        "display_title": BeautifulSoup(display, "html.parser").get_text(" ", strip=True),
        "file": f"pages/{filename}",
        "source": source_url,
        "translated_blocks": done,
        "total_blocks": total,
        "updated_at": utcnow(),
        "model": MODEL,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "10")))
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing. Add it as a GitHub Actions repository secret.")

    DATA.mkdir(exist_ok=True)
    PAGES.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    client = OpenAI(api_key=api_key)

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
    candidates.sort(key=lambda p: (failures.get(p["title"], {}).get("attempts", 0), order[p["title"]]))
    selected = candidates[:max(1, args.batch_size)]

    print(f"Total={len(titles)} Completed={len(completed)} Selected={len(selected)} Model={MODEL}")

    for n, page in enumerate(selected, 1):
        title = page["title"]
        print(f"[{n}/{len(selected)}] {title}")
        try:
            manifest[title] = translate_page(session, client, page)
            failures.pop(title, None)
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
        "model": MODEL,
        "batch_size": args.batch_size,
    }
    save_json(PROGRESS, progress)
    print(json.dumps(progress, ensure_ascii=False))

if __name__ == "__main__":
    main()
