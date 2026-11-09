#!/usr/bin/env python3
"""Morning digest: fetch feeds, rank with an LLM against taste.md, render public/index.html.

Usage:
    python digest.py            # normal run (uses ANTHROPIC_API_KEY if set, writes state/)
    python digest.py --no-llm   # force the chronological fallback
    python digest.py --mock     # offline: synthetic feeds + fake LLM response, no state writes
    python digest.py --dry-run  # real feeds, but don't write state/ or touch GitHub issues
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import logging
import os
import re
import sys
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import yaml
from dateutil import parser as dateparser
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
SEEN_PATH = STATE / "seen.json"
PAGES_PATH = STATE / "pages.json"
FEEDBACK_PATH = STATE / "feedback.jsonl"
LEARNED_PATH = STATE / "learned_taste.md"
META_PATH = STATE / "meta.json"
LIBRARY_PATH = STATE / "library.json"  # {url: {title, source, category, why_it_matters, list, added}}
LIBRARY_LISTS = ("later", "saved")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "48"))
EXPANDED_LOOKBACK_HOURS = 24 * 7  # used when the normal window is thin
MIN_CANDIDATES = 15
MAX_CANDIDATES = 500
MAX_PER_SOURCE = 5  # newest N per source, so busy feeds can't crowd out quiet ones
TOP_N = 5
WORDS_PER_MINUTE = 238
USER_AGENT = "Mozilla/5.0 (compatible; morning-digest/1.0; +https://github.com)"  # GitHub API only
# Some sites (Substack especially) refuse obvious bots from cloud IPs, so feeds and
# articles are fetched with the headers a desktop Chrome would send.
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/html;q=0.8, */*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}
FEEDBACK_ISSUE_PREFIX = "Digest feedback"
SEEN_RETENTION_DAYS = 30

log = logging.getLogger("digest")


# --------------------------------------------------------------------------- utils

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text or "")
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def http_get(url: str, timeout: int = 15, headers: dict | None = None) -> bytes:
    # Try browser-like headers first; some sites block those instead, so on a 403
    # retry once as a plain, honest bot.
    attempts = [BROWSER_HEADERS, {"User-Agent": USER_AGENT}]
    for i, base in enumerate(attempts):
        req = urllib.request.Request(url, headers={**base, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 403 or i == len(attempts) - 1:
                raise


def check_feeds(feeds: list[dict]) -> int:
    """Fetch every source once and report which ones work from this machine."""
    def probe(feed):
        try:
            return feed, len(fetch_page_source(feed) if feed.get("type") == "page"
                             else parse_feed(feed, http_get(feed["url"]))), ""
        except Exception as exc:
            return feed, 0, str(exc)[:80]

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(probe, feeds))
    bad = [(f, err or "no dated entries") for f, n, err in results if not n]
    for f, why in bad:
        print(f"FAIL  {f['name']:<45} {why}")
    print(f"\n{len(feeds) - len(bad)}/{len(feeds)} sources OK")
    return 0


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    out = []
    for line in FEEDBACK_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping bad feedback line: %s", line[:80])
    return latest_vote_per_article(out)


def latest_vote_per_article(feedback: list[dict]) -> list[dict]:
    """If an article was rated more than once (e.g. feedback sent twice), keep only the latest."""
    latest = {f["url"]: f for f in feedback}
    return [f for f in feedback if latest[f["url"]] is f]


# --------------------------------------------------------------------------- feeds

def load_feeds(path: Path = ROOT / "feeds.yaml") -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    feeds = data.get("feeds", data if isinstance(data, list) else [])
    return [f for f in feeds if f.get("name") and f.get("url")]


def entry_datetime(entry) -> dt.datetime | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        t = entry.get(key)
        if t:
            return dt.datetime(*t[:6], tzinfo=dt.timezone.utc)
    for key in ("published", "updated", "created"):
        if entry.get(key):
            try:
                d = dateparser.parse(entry[key])
                return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
            except (ValueError, OverflowError):
                pass
    return None


def parse_feed(feed: dict, raw: bytes) -> list[dict]:
    parsed = feedparser.parse(raw)
    items = []
    for e in parsed.entries:
        link = e.get("link")
        published = entry_datetime(e)
        if not link or not published:
            continue
        summary = strip_html(e.get("summary", ""))
        full = ""
        if e.get("content"):
            full = strip_html(" ".join(c.get("value", "") for c in e.content))
        items.append({
            "title": strip_html(e.get("title", "")) or "(untitled)",
            "url": link,
            "source": feed["name"],
            "category": feed.get("category", ""),
            "author": strip_html(e.get("author", "")),
            "published": published.isoformat(),
            "summary": summary or full[:1000],
            # Full-text word count when the feed ships content:encoded (0 = unknown).
            "words": word_count(full) if word_count(full) > word_count(summary) else 0,
        })
    return items


def meta_tag(page: str, *names: str) -> str:
    for n in names:
        n = re.escape(n)
        m = (re.search(rf'<meta[^>]+(?:property|name|itemprop)=["\']{n}["\'][^>]*content=["\']([^"\']*)', page, re.I)
             or re.search(rf'<meta[^>]+content=["\']([^"\']*)["\'][^>]*(?:property|name|itemprop)=["\']{n}["\']', page, re.I))
        if m:
            return html.unescape(m.group(1)).strip()
    return ""


def scrape_post(url: str) -> dict:
    page = http_get(url).decode("utf-8", "ignore")
    title = meta_tag(page, "og:title", "twitter:title")
    if not title:
        m = re.search(r"(?is)<title>(.*?)</title>", page)
        title = strip_html(m.group(1)) if m else url
    published = meta_tag(page, "article:published_time", "datePublished", "date", "pubdate")
    m = re.search(r"(?is)<article.*?</article>", page)
    return {
        "title": title,
        "summary": meta_tag(page, "og:description", "description", "twitter:description"),
        "author": meta_tag(page, "author", "article:author"),
        "published": published,
        "words": word_count(strip_html(m.group(0) if m else page)),
    }


# Blog index pages watched for sources without a usable feed: {index_url: {post_url: meta}}.
PAGE_CACHE: dict[str, dict] = {}
EPOCH = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)


def fetch_page_source(feed: dict) -> list[dict]:
    """For blogs with no usable RSS: find post links on the index page and scrape each new one once."""
    base = feed["url"]
    page = http_get(base).decode("utf-8", "ignore")
    pattern = re.compile(feed.get("link_pattern", "."))
    host = urllib.parse.urlparse(base).netloc
    links: list[str] = []
    for href in re.findall(r'href=["\']([^"\'#]+)', page):
        url = urllib.parse.urljoin(base, html.unescape(href))
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc == host and pattern.search(parsed.path) and url.rstrip("/") != base.rstrip("/") and url not in links:
            links.append(url)

    known = PAGE_CACHE.setdefault(base, {})
    first_crawl = not known
    items = []
    for url in links[:40]:
        meta = known.get(url)
        if meta is None:
            try:
                meta = scrape_post(url)
            except Exception as exc:
                log.warning("couldn't scrape %s (%s)", url, exc)
                continue
            published = None
            if meta["published"]:
                try:
                    published = dateparser.parse(meta["published"])
                    published = published if published.tzinfo else published.replace(tzinfo=dt.timezone.utc)
                except (ValueError, OverflowError):
                    published = None
            # Undated posts count as "published" when first noticed, except on the very
            # first crawl, so adding a blog doesn't dump its whole back catalog on you.
            meta["published"] = (published or (EPOCH if first_crawl else now_utc())).isoformat()
            known[url] = meta
        items.append({
            "title": meta["title"], "url": url, "source": feed["name"],
            "category": feed.get("category", ""), "author": meta.get("author", ""),
            "published": meta["published"], "summary": meta.get("summary", ""), "words": meta.get("words", 0),
        })
    return items


def fetch_feed(feed: dict) -> list[dict]:
    try:
        if feed.get("type") == "page":
            return fetch_page_source(feed)
        return parse_feed(feed, http_get(feed["url"]))
    except Exception as exc:  # one bad feed shouldn't sink the digest
        log.warning("feed failed: %s (%s)", feed["name"], exc)
        return []


def gather_candidates(feeds: list[dict], seen: dict, fetcher=fetch_feed) -> tuple[list[dict], int]:
    with ThreadPoolExecutor(max_workers=16) as pool:
        all_items = [i for items in pool.map(fetcher, feeds) for i in items]

    unseen, urls = [], set()
    for item in all_items:
        if item["url"] in seen or item["url"] in urls:
            continue
        urls.add(item["url"])
        unseen.append(item)

    def within(hours):
        cutoff = now_utc() - dt.timedelta(hours=hours)
        return [i for i in unseen if dt.datetime.fromisoformat(i["published"]) >= cutoff]

    window = LOOKBACK_HOURS
    candidates = within(window)
    if len(candidates) < MIN_CANDIDATES:
        window = EXPANDED_LOOKBACK_HOURS
        candidates = within(window)
    candidates.sort(key=lambda i: i["published"], reverse=True)
    per_source: dict[str, int] = {}
    capped = []
    for c in candidates:
        per_source[c["source"]] = per_source.get(c["source"], 0) + 1
        if per_source[c["source"]] <= MAX_PER_SOURCE:
            capped.append(c)
    candidates = capped
    log.info("%d items fetched, %d candidates in last %dh", len(all_items), len(candidates), window)
    return candidates[:MAX_CANDIDATES], window


# --------------------------------------------------------------------------- feedback

def github_api(method: str, path: str, body: dict | None = None):
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return None
    url = f"https://api.github.com/repos/{repo}{path}"
    req = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read() or b"null")


def parse_feedback_issue(body: str) -> dict | None:
    m = re.search(r"```json\s*(\{.*?\})\s*```", body or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("votes", []), list) or not isinstance(data.get("library", []), list):
        return None
    return data if ("votes" in data or "library" in data) else None


def ingest_feedback_issues() -> tuple[list[dict], list[dict]]:
    """Pull 'Digest feedback' issues opened by the repo owner, close them.
    Returns (votes, library operations), oldest issue first."""
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "").lower()
    try:
        issues = github_api("GET", "/issues?state=open&per_page=50") or []
    except Exception as exc:
        log.warning("couldn't list feedback issues: %s", exc)
        return [], []

    new, library_ops = [], []
    for issue in sorted(issues, key=lambda i: i["number"]):
        if "pull_request" in issue or not issue.get("title", "").startswith(FEEDBACK_ISSUE_PREFIX):
            continue
        # Public repo: anyone can open an issue. Only trust the owner's.
        if issue["user"]["login"].lower() != owner and issue.get("author_association") != "OWNER":
            log.info("ignoring feedback issue #%s from %s", issue["number"], issue["user"]["login"])
            continue
        data = parse_feedback_issue(issue.get("body", ""))
        if not data:
            log.warning("feedback issue #%s has no parseable JSON", issue["number"])
            continue
        for v in data.get("votes", []):
            if v.get("vote") not in ("up", "down") or not v.get("url"):
                continue
            new.append({
                "date": data.get("date", ""),
                "issue": issue["number"],
                "url": v["url"],
                "title": str(v.get("title", ""))[:300],
                "source": str(v.get("source", ""))[:100],
                "vote": v["vote"],
                "tags": [str(t)[:40] for t in v.get("tags", [])][:10],
                "note": str(v.get("note", ""))[:500],
                "wildcard": bool(v.get("wildcard")),
            })
        for item in data.get("library", []):
            if isinstance(item, dict) and item.get("url"):
                library_ops.append({**item, "date": data.get("date", "")})
        try:
            n_votes, n_lib = len(data.get("votes", [])), len(data.get("library", []))
            github_api("POST", f"/issues/{issue['number']}/comments",
                       {"body": f"Ingested {n_votes} vote(s) and {n_lib} library change(s)."})
            github_api("PATCH", f"/issues/{issue['number']}", {"state": "closed"})
        except Exception as exc:
            log.warning("couldn't close issue #%s: %s", issue["number"], exc)
    log.info("ingested %d feedback votes, %d library changes", len(new), len(library_ops))
    return new, library_ops


def apply_library_ops(library: dict, ops: list[dict]) -> dict:
    """Add, move or remove reading-list items. list = "later" | "saved"; anything else removes."""
    for op in ops:
        url = str(op["url"])
        if op.get("list") in LIBRARY_LISTS:
            prev = library.get(url, {})
            library[url] = {
                "url": url,
                "title": str(op.get("title") or prev.get("title") or url)[:300],
                "source": str(op.get("source") or prev.get("source") or "")[:100],
                "category": str(op.get("category") or prev.get("category") or "")[:40],
                "why_it_matters": str(op.get("why_it_matters") or prev.get("why_it_matters") or "")[:400],
                "list": op["list"],
                "added": prev.get("added") or op.get("date") or now_utc().strftime("%Y-%m-%d"),
            }
        else:
            library.pop(url, None)
    return library


def format_library_examples(library: dict, limit: int = 20) -> str:
    items = sorted(library.values(), key=lambda i: i["added"])[-limit:]
    label = {"saved": "SAVED to keep", "later": "SAVED to read later"}
    return "\n".join(f'- {label[i["list"]]}: "{i["title"]}" ({i["source"]})' for i in items)


def source_stats(feedback: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for f in feedback:
        s = stats.setdefault(f["source"], {"up": 0, "down": 0})
        s[f["vote"]] += 1
    return stats


def format_feedback_examples(feedback: list[dict], limit: int = 25) -> str:
    lines = []
    for f in feedback[-limit:]:
        mark = "LIKED" if f["vote"] == "up" else "DISLIKED"
        why = ", ".join(f.get("tags", []))
        note = f' — "{f["note"]}"' if f.get("note") else ""
        wild = " [was the wildcard]" if f.get("wildcard") else ""
        lines.append(f'- {mark}: "{f["title"]}" ({f["source"]}){wild}' + (f" because: {why}" if why else "") + note)
    return "\n".join(lines)


# --------------------------------------------------------------------------- LLM

EDITOR_INSTRUCTIONS = """You are the editor of a one-reader morning digest. Your job is to pick the \
{n} pieces most worth this reader's time today from the candidates below.

Judge every candidate on:
- Technical depth or intellectual depth: does it actually explain, argue, or show something?
- Novel insight: would a well-read engineer learn something they don't already know?
- Quality of writing: a strong voice and a real idea beat a competent summary.

Strictly reject: product announcements and launches, funding/earnings news, superficial AI hype, \
listicles, surface-level news, clickbait, link roundups, and pieces that only summarize other work. \
The summaries you see are excerpts; judge the piece they describe, not the excerpt's polish.

Pick exactly {n}: {core} that fit the reader's interests, plus exactly 1 WILDCARD — the single best \
piece from OUTSIDE their stated interests (any field, any style) that they'd likely never have \
looked for but would be glad they read. Mark it wildcard=true. Favor depth over recency.

For each pick, write why_it_matters: ONE line (max ~25 words) — the concrete reason to read it, \
naming the key idea or takeaway. No hype words, no "this article".
Score each 1-10 for how strongly you'd recommend it to this reader. Order picks best first, \
wildcard last.

<reader_taste>
{taste}
</reader_taste>
{learned}{examples}{sources}"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                    "why_it_matters": {"type": "string"},
                    "wildcard": {"type": "boolean"},
                },
                "required": ["id", "score", "why_it_matters", "wildcard"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["picks"],
    "additionalProperties": False,
}


def build_system_prompt(taste: str, learned: str, feedback: list[dict], library: dict | None = None) -> str:
    learned_block = f"\n<learned_from_feedback>\n{learned.strip()}\n</learned_from_feedback>\n" if learned.strip() else ""
    examples = "\n".join(x for x in (format_feedback_examples(feedback), format_library_examples(library or {})) if x)
    examples_block = (
        "\n<recent_reactions>\nThe reader's recent reactions to past picks. Use the reasons, "
        "not just the votes. Saving a piece is a strong signal of interest, even unread."
        f"\n{examples}\n</recent_reactions>\n" if examples else ""
    )
    stats = {k: v for k, v in source_stats(feedback).items() if v["up"] + v["down"] >= 3}
    sources_block = ""
    if stats:
        rows = "\n".join(f"- {k}: {v['up']} liked, {v['down']} disliked" for k, v in sorted(stats.items()))
        sources_block = f"\n<source_track_record>\n{rows}\n</source_track_record>\n"
    return EDITOR_INSTRUCTIONS.format(
        n=TOP_N, core=TOP_N - 1, taste=taste.strip(),
        learned=learned_block, examples=examples_block, sources=sources_block,
    )


def build_user_prompt(candidates: list[dict]) -> str:
    rows = []
    for i, c in enumerate(candidates):
        length = f"~{c['words']} words" if c["words"] else "length unknown"
        rows.append(
            f'<candidate id="{i}">\n'
            f"title: {c['title']}\nsource: {c['source']} ({c['category'] or 'general'})\n"
            f"author: {c['author'] or 'unknown'}\npublished: {c['published'][:10]}\n{length}\n"
            f"excerpt: {c['summary'][:600]}\n</candidate>"
        )
    return f"{len(candidates)} candidates:\n\n" + "\n\n".join(rows) + \
        f"\n\nReturn your {TOP_N} picks as JSON by candidate id."


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_llm(system: str, user: str, schema: dict | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    kwargs = {"output_config": {"format": {"type": "json_schema", "schema": schema}}} if schema else {}
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        **kwargs,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("model hit max_tokens")
    return next(b.text for b in resp.content if b.type == "text")


def validate_picks(raw: str, candidates: list[dict]) -> list[dict]:
    """Map the model's id-based picks back onto real candidates. Never trust model-written URLs."""
    data = json.loads(raw)
    picks, used = [], set()
    for p in data.get("picks", []):
        try:
            idx = int(p["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx in used or not 0 <= idx < len(candidates):
            continue
        used.add(idx)
        c = candidates[idx]
        picks.append({
            **c,
            "score": max(1, min(10, int(p.get("score", 5)))),
            "why_it_matters": str(p.get("why_it_matters", "")).strip(),
            "wildcard": bool(p.get("wildcard")),
        })
    # Keep at most one wildcard, and put it last.
    wild = [p for p in picks if p["wildcard"]][:1]
    core = [p for p in picks if not p["wildcard"]]
    for p in core:
        p["wildcard"] = False
    return (core[: TOP_N - len(wild)] + wild)[:TOP_N]


def rank(candidates: list[dict], taste: str, learned: str, feedback: list[dict],
         use_llm: bool, llm=call_llm, library: dict | None = None) -> tuple[list[dict], str]:
    """Returns (picks, label describing how they were chosen)."""
    if use_llm and candidates:
        try:
            raw = llm(build_system_prompt(taste, learned, feedback, library),
                      build_user_prompt(candidates), OUTPUT_SCHEMA)
            picks = validate_picks(raw, candidates)
            if picks:
                if len(picks) < TOP_N:  # top up with newest unpicked items
                    picked = {p["url"] for p in picks}
                    picks += fallback_picks([c for c in candidates if c["url"] not in picked], TOP_N - len(picks))
                    picks.sort(key=lambda p: p["wildcard"])  # stable: wildcard stays last
                return picks, f"Ranked by {ANTHROPIC_MODEL}"
            log.warning("LLM returned no usable picks; falling back")
        except Exception as exc:
            log.warning("LLM ranking failed: %s; falling back", exc)
    reason = "no API key set" if not use_llm else "ranking failed"
    return fallback_picks(candidates, TOP_N), f"Newest items ({reason})"


def fallback_picks(candidates: list[dict], n: int) -> list[dict]:
    out = []
    for c in candidates[:n]:
        summary = c["summary"]
        first = re.split(r"(?<=[.!?])\s", summary, maxsplit=1)[0] if summary else ""
        out.append({**c, "score": None, "wildcard": False,
                    "why_it_matters": (first[:220] + "…") if len(first) > 220 else (first or "No summary available.")})
    return out


# --------------------------------------------------------------------------- learned taste

LEARN_PROMPT = """Below is the reader's stated taste profile and their reactions to past digest picks \
(liked/disliked, with reason tags and notes). Write a short "learned preferences" note — at most 8 \
bullet points — capturing patterns in the feedback that refine or go beyond the stated profile: \
topics, sources, lengths, styles, and what kinds of wildcard picks land. Be specific and concrete. \
Don't restate the profile; only add what the feedback teaches. If feedback contradicts the profile, \
say so. Plain markdown bullets, no preamble."""


def maybe_update_learned_taste(use_llm: bool, taste: str, feedback: list[dict], meta: dict,
                               llm=call_llm) -> bool:
    last = meta.get("learned_updated")
    count_at_last = meta.get("learned_feedback_count", 0)
    due = not last or (now_utc() - dt.datetime.fromisoformat(last)).days >= 7
    if not use_llm or not due or len(feedback) - count_at_last < 5:
        return False
    try:
        text = llm(LEARN_PROMPT,
                   f"<profile>\n{taste}\n</profile>\n\n<reactions>\n{format_feedback_examples(feedback, 200)}\n</reactions>")
    except Exception as exc:
        log.warning("learned-taste update failed: %s", exc)
        return False
    stamp = now_utc().strftime("%Y-%m-%d")
    LEARNED_PATH.write_text(
        f"# Learned from your feedback\n\n_Auto-generated weekly from {len(feedback)} votes. "
        f"Last updated {stamp}. Edit taste.md, not this file._\n\n{text.strip()}\n"
    )
    meta.update(learned_updated=now_utc().isoformat(), learned_feedback_count=len(feedback))
    log.info("learned taste updated")
    return True


# --------------------------------------------------------------------------- reading time

def add_reading_times(picks: list[dict], fetch=http_get) -> None:
    def estimate(p):
        words = p.get("words") or 0
        if words < 300:
            try:
                page = fetch(p["url"], timeout=12).decode("utf-8", "ignore")
                m = re.search(r"(?is)<article.*?</article>", page)
                words = max(words, word_count(strip_html(m.group(0) if m else page)))
            except Exception:
                pass
        p["minutes"] = max(1, round(words / WORDS_PER_MINUTE)) if words >= 150 else None

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(estimate, picks))


# --------------------------------------------------------------------------- render

CATEGORY_LABELS = {
    "hardware": "Hardware", "systems": "Systems", "ai-ml": "AI / ML", "math-physics": "Math & Physics",
    "explainers": "Explainer", "science-essays": "Science", "essays": "Essay", "literary": "Literary",
    "fiction": "Fiction", "history": "History", "wildcard": "Curiosities",
}


def render(picks: list[dict], method: str, window_hours: int, n_candidates: int, out_dir: Path,
           library: dict | None = None) -> Path:
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape(["html", "j2"]))
    env.filters["host"] = lambda u: urllib.parse.urlparse(u).netloc.removeprefix("www.")
    env.filters["catlabel"] = lambda c: CATEGORY_LABELS.get(c or "", (c or "Reading").replace("-", " ").title())
    picks = sorted(picks, key=lambda p: p.get("wildcard", False))  # wildcard last; matches the page's indices
    today = now_utc().astimezone(ZoneInfo("America/New_York"))
    repo = os.environ.get("GITHUB_REPOSITORY") or os.environ.get("DIGEST_REPO", "")
    library_items = sorted((library or {}).values(), key=lambda i: i["added"], reverse=True)
    common = dict(
        date_iso=today.strftime("%Y-%m-%d"),
        repo=repo,
        issue_prefix=FEEDBACK_ISSUE_PREFIX,
        library_js={i["url"]: i["list"] for i in library_items},
        library_items=library_items,
        category_labels=CATEGORY_LABELS,
    )
    page = env.get_template("index.html.j2").render(
        **common,
        picks=picks,
        picks_js=[{k: p.get(k) for k in ("url", "title", "source", "wildcard", "category", "why_it_matters")}
                  for p in picks],
        date_human=today.strftime("%A, %B %-d, %Y"),
        date_short=today.strftime("%b %-d"),
        greeting="Good morning",  # it's a morning paper, whenever GitHub gets around to printing it
        method=method,
        window_hours=window_hours,
        n_candidates=n_candidates,
        total_minutes=sum(p.get("minutes") or 0 for p in picks),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(page)
    (out_dir / "library.html").write_text(env.get_template("library.html.j2").render(**common))
    (out_dir / ".nojekyll").write_text("")
    return out


# --------------------------------------------------------------------------- mock mode

MOCK_SAMPLES = [
    ("Why HBM bandwidth, not FLOPs, bounds LLM inference", "A deep look at memory-bound decode on modern accelerators.", 2400, "Sam Rivera"),
    ("Startup X raises $40M for AI-native spreadsheets", "The company announced its Series B today.", 300, ""),
    ("The strange arithmetic of sphere packing in high dimensions", "Mathematicians found a new lower bound.", 3100, "Lena Okafor"),
    ("On Self-Reliance, again", "An essay on attention and solitude in the age of feeds.", 4200, "Thomas Hale"),
    ("10 AI tools you need this week", "Our roundup of the hottest launches.", 600, ""),
    ("The forgotten cartographers of the Ottoman court", "A narrative history of mapmaking.", 5200, "Defne Arslan"),
    ("What induction heads tell us about in-context learning", "Circuits, revisited.", 3600, "Priya Natarajan"),
    ("How concrete got cheap", "The industrial history of cement kilns.", 2900, ""),
]


def mock_fetcher(feed: dict) -> list[dict]:
    """Synthetic feed so the whole pipeline can run offline."""
    t = now_utc()
    offset = sum(map(ord, feed["name"]))  # stagger dates so every sample shows up somewhere
    rss = "".join(
        f"<item><title>{html.escape(title)}</title><author>{html.escape(author)}</author>"
        f"<link>https://{feed['url'].split('/')[2]}/posts/{i}</link>"
        f"<pubDate>{(t - dt.timedelta(hours=(i * 5 + offset) % 44 + 1)).strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>"
        f"<description>{html.escape(desc)}</description>"
        f"<content:encoded>{html.escape(('word ' * words))}</content:encoded></item>"
        for i, (title, desc, words, author) in enumerate(MOCK_SAMPLES)
    )
    raw = (f'<?xml version="1.0"?><rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
           f"<channel><title>{html.escape(feed['name'])}</title>{rss}</channel></rss>").encode()
    return parse_feed(feed, raw)


MOCK_REASONS = {
    "Why HBM": "Shows, with real roofline numbers, why decode speed tracks memory bandwidth, and what that means for the next accelerator generation.",
    "sphere packing": "A new lower bound in high dimensions, explained through a surprisingly simple random construction.",
    "Self-Reliance": "An Emersonian argument that attention is the last thing you truly own, written with real bite.",
    "induction heads": "Revisits the circuit behind in-context learning and finds it does more than copying.",
    "cartographers": "A narrative history of how maps were made, and argued over, at the Ottoman court.",
    "concrete": "How kiln engineering, not chemistry, made cement cheap, and why that matters for housing today.",
}


def mock_llm(system, user, schema=None) -> str:
    if schema is None:
        return "- Likes memory-hierarchy deep dives\n- Dislikes funding news"
    ids = [int(m) for m in re.findall(r'<candidate id="(\d+)">', user)]
    titles = dict(zip(ids, re.findall(r"title: (.*)", user)))
    order = ["Why HBM", "sphere packing", "Self-Reliance", "induction heads", "concrete"]
    picks = []
    for k, key in enumerate(order):
        i = next(i for i in ids if key in titles[i])
        picks.append({"id": i, "score": 9 - k // 2, "why_it_matters": MOCK_REASONS[key], "wildcard": key == "concrete"})
    picks.append({"id": 99999, "score": 10, "why_it_matters": "hallucinated id, must be dropped", "wildcard": False})
    return json.dumps({"picks": picks})


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="offline run with synthetic feeds and a fake LLM")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM and use the chronological fallback")
    ap.add_argument("--dry-run", action="store_true", help="don't write state/ or touch GitHub issues")
    ap.add_argument("--out", default="public", help="output directory (default: public)")
    ap.add_argument("--check-feeds", action="store_true", help="only report which sources can be fetched")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.check_feeds:
        return check_feeds(load_feeds())
    dry = args.dry_run or args.mock
    feeds = load_feeds()
    seen: dict = {} if args.mock else read_json(SEEN_PATH, {})
    meta: dict = read_json(META_PATH, {})
    taste = (ROOT / "taste.md").read_text()
    PAGE_CACHE.update(read_json(PAGES_PATH, {}))

    feedback = read_feedback()
    library: dict = read_json(LIBRARY_PATH, {})
    if not dry:
        new_votes, library_ops = ingest_feedback_issues()
        if library_ops:
            STATE.mkdir(exist_ok=True)
            apply_library_ops(library, library_ops)
            LIBRARY_PATH.write_text(json.dumps(library, indent=1, sort_keys=True) + "\n")
        if new_votes:
            STATE.mkdir(exist_ok=True)
            with FEEDBACK_PATH.open("a") as fh:
                for v in new_votes:
                    fh.write(json.dumps(v) + "\n")
            feedback = latest_vote_per_article(feedback + new_votes)

    if args.no_llm:
        use_llm, llm = False, call_llm
    elif args.mock:
        use_llm, llm = True, mock_llm
    else:
        use_llm, llm = has_api_key(), call_llm
    log.info("ranking with: %s", ANTHROPIC_MODEL if use_llm else "none (chronological fallback)")

    if not dry:
        STATE.mkdir(exist_ok=True)
        maybe_update_learned_taste(use_llm, taste, feedback, meta, llm=llm)
    learned = LEARNED_PATH.read_text() if LEARNED_PATH.exists() else ""
    if "Nothing learned yet" in learned:
        learned = ""

    candidates, window = gather_candidates(feeds, seen, fetcher=mock_fetcher if args.mock else fetch_feed)
    picks, method = rank(candidates, taste, learned, feedback, use_llm, llm=llm, library=library)
    if args.mock:
        for p in picks:
            p["minutes"] = max(1, round((p.get("words") or 0) / WORDS_PER_MINUTE)) or None
    else:
        add_reading_times(picks)

    out = render(picks, method, window, len(candidates), ROOT / args.out, library=library)
    log.info("wrote %s (%d picks; %s)", out, len(picks), method)

    if not dry:
        today = now_utc().strftime("%Y-%m-%d")
        for p in picks:
            seen[p["url"]] = today
        cutoff = (now_utc() - dt.timedelta(days=SEEN_RETENTION_DAYS)).strftime("%Y-%m-%d")
        seen = {u: d for u, d in seen.items() if d >= cutoff}
        SEEN_PATH.write_text(json.dumps(seen, indent=1, sort_keys=True) + "\n")
        PAGES_PATH.write_text(json.dumps(PAGE_CACHE, indent=1, sort_keys=True) + "\n")
        meta["last_built"] = now_utc().astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        META_PATH.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
