"""PubMed E-utilities wrapper for the literature-driven auto-research agent.

Provides two tools the LLM can call:
    pubmed_search(query, max_results) -> [{"pmid","title","year"}]
    fetch_abstract(pmid) -> {"pmid","title","abstract","year","authors"}

NCBI rate limit (no API key): 3 req/sec. We enforce 0.5s between calls.
"""
from __future__ import annotations
import urllib.request, urllib.parse, json, re, time, threading
from typing import List, Dict

_LOCK = threading.Lock()
_LAST_CALL = [0.0]
_MIN_INTERVAL = 0.5  # seconds

def _throttle():
    with _LOCK:
        elapsed = time.time() - _LAST_CALL[0]
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _LAST_CALL[0] = time.time()

def _http_get(url: str, timeout: int = 15) -> str:
    _throttle()
    req = urllib.request.Request(url, headers={
        "User-Agent": "nmr-research-agent/1.0"
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

def pubmed_search(query: str, max_results: int = 10) -> List[Dict]:
    """Search PubMed and return basic metadata (PMIDs + titles)."""
    if not query.strip():
        return []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": str(max_results),
              "retmode": "json", "sort": "relevance"}
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        raw = _http_get(url)
        data = json.loads(raw)
        pmids = data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        return [{"error": f"pubmed_search failed: {e}"}]
    if not pmids:
        return []
    # Get titles via esummary
    summary_url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                   f"?db=pubmed&id={','.join(pmids)}&retmode=json")
    try:
        raw = _http_get(summary_url)
        summary = json.loads(raw).get("result", {})
        out = []
        for pmid in pmids:
            entry = summary.get(pmid, {})
            out.append({
                "pmid": pmid,
                "title": entry.get("title", ""),
                "year": entry.get("pubdate", "")[:4],
                "journal": entry.get("source", ""),
            })
        return out
    except Exception:
        return [{"pmid": p, "title": "", "year": "", "journal": ""} for p in pmids]

def fetch_abstract(pmid: str) -> Dict:
    """Fetch full abstract for a single PMID."""
    pmid = str(pmid).strip()
    if not pmid.isdigit():
        return {"error": f"invalid pmid: {pmid}"}
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           f"?db=pubmed&id={pmid}&retmode=xml")
    try:
        xml = _http_get(url, timeout=20)
    except Exception as e:
        return {"error": f"fetch failed: {e}", "pmid": pmid}

    title = re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", xml, re.S)
    title_text = re.sub(r"<[^>]+>", "", title.group(1)) if title else ""

    # Abstract may have multiple <AbstractText> sections
    abs_sections = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, re.S)
    abstract_text = " ".join(re.sub(r"<[^>]+>", "", s) for s in abs_sections)

    year = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.S)
    year_text = year.group(1) if year else ""

    journal = re.search(r"<Journal>.*?<Title>(.*?)</Title>", xml, re.S)
    journal_text = journal.group(1) if journal else ""

    return {
        "pmid": pmid,
        "title": title_text.strip(),
        "abstract": abstract_text.strip(),
        "year": year_text,
        "journal": journal_text,
    }

if __name__ == "__main__":
    # Smoke test
    res = pubmed_search("lactate pyruvate ratio serum NMR", max_results=3)
    for r in res: print(r)
    if res and "pmid" in res[0]:
        print("\n--- Abstract ---")
        a = fetch_abstract(res[0]["pmid"])
        print(a["title"])
        print(a["abstract"][:400])
