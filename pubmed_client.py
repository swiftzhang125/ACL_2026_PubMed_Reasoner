"""
PubMed E-utilities API wrapper for searching and fetching articles.
"""

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Free tier: 3 requests/sec
MIN_REQUEST_INTERVAL = 0.34


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str


class PubMedClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def search(self, query: str, max_results: int = 20) -> list[str]:
        """Search PubMed and return a list of PMIDs."""
        self._rate_limit()

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "sort": "relevance",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        resp = requests.get(ESEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        return [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]

    def fetch_articles(self, pmids: list[str]) -> list[Article]:
        """Fetch title and abstract for a list of PMIDs."""
        if not pmids:
            return []

        self._rate_limit()

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        resp = requests.get(EFETCH_URL, params=params, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)

        articles = []
        for art in root.findall(".//PubmedArticle"):
            pmid_elem = art.find(".//PMID")
            title_elem = art.find(".//ArticleTitle")
            abstract_parts = art.findall(".//AbstractText")

            pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""
            title = title_elem.text if title_elem is not None and title_elem.text else ""

            abstract = " ".join(
                part.text for part in abstract_parts if part.text
            )

            articles.append(Article(pmid=pmid, title=title, abstract=abstract))

        return articles

    def search_and_fetch(self, query: str, max_results: int = 20) -> list[Article]:
        """Search PubMed and fetch article details in one call."""
        pmids = self.search(query, max_results=max_results)
        if not pmids:
            return []
        return self.fetch_articles(pmids)