"""Internet search module for the package."""

import asyncio
import re

import numpy as np
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from duckduckgo_search import AsyncDDGS
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

from . import GeneralDefinitions
from .general_utils import retry


def cosine_similarity_sentences(sentence1, sentence2):
    """Compute the cosine similarity between two sentences."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]


def element_is_visible(element):
    """Return True if the element is visible."""
    tags_to_exclude = [
        "[document]",
        "head",
        "header",
        "html",
        "input",
        "meta",
        "noscript",
        "script",
        "style",
        "style",
        "title",
    ]
    if element.parent.name in tags_to_exclude or isinstance(element, Comment):
        return False
    return True


def extract_text_from_html(body):
    """Extract the text from an HTML document."""
    soup = BeautifulSoup(body, "html.parser")

    page_has_captcha = soup.find("div", id="recaptcha") is not None
    if page_has_captcha:
        return ""

    texts = soup.find_all(string=True)
    visible_texts = filter(element_is_visible, texts)
    return " ".join(t.strip() for t in visible_texts if t.strip())


def find_whole_word_index(my_string, my_substring):
    """Find the index of a substring in a string, but only if it is a whole word match."""
    pattern = re.compile(r"\b{}\b".format(re.escape(my_substring)))
    match = pattern.search(my_string)

    if match:
        return match.start()
    return -1  # Substring not found


async def async_raw_websearch(
    query: str,
    max_results: int = 5,
    region: str = GeneralDefinitions.IPINFO["country_name"],
):
    """Search the web using DuckDuckGo Search API."""
    async with AsyncDDGS(proxies=None) as addgs:
        results = await addgs.text(
            keywords=query,
            region=region,
            max_results=max_results,
            backend="html",
        )
        return results


def raw_websearch(
    query: str,
    max_results: int = 5,
    region: str = GeneralDefinitions.IPINFO["country_name"],
):
    """Search the web using DuckDuckGo Search API."""
    raw_results = asyncio.run(
        async_raw_websearch(query=query, max_results=max_results, region=region)
    )
    raw_results = raw_results or []

    results = []
    for result in raw_results:
        if not isinstance(result, dict):
            logger.error("Expected a `dict`, got type {}: {}", type(result), result)
            results.append({})
            continue

        if result.get("body") is None:
            continue

        try:
            response = requests.get(result["href"], allow_redirects=False, timeout=10)
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            continue
        else:
            content_type = response.headers.get("content-type")
            if (not content_type) or ("text/html" not in content_type):
                continue
            html = unidecode(extract_text_from_html(response.text))

        summary = unidecode(result["body"])
        relevance = cosine_similarity_sentences(query.lower(), summary.lower())

        relevance_threshold = 1e-2
        if relevance < relevance_threshold:
            continue

        new_results = {
            "href": result["href"],
            "summary": summary,
            "detailed": html,
            "relevance": relevance,
        }
        results.append(new_results)
    return results


@retry(error_msg="Error performing web search")
def websearch(query, **kwargs):
    """Search the web using DuckDuckGo Search API."""
    raw_results = raw_websearch(query, **kwargs)
    raw_results = iter(
        sorted(raw_results, key=lambda x: x.get("relevance", 0.0), reverse=True)
    )
    min_relevant_keyword_length = 4
    min_n_words = 40

    for result in raw_results:
        html = result.get("detailed", "")

        index_first_query_word_to_appear = np.inf
        for word in unidecode(query).split():
            if len(word) < min_relevant_keyword_length:
                continue
            index = find_whole_word_index(html.lower(), word.lower())
            if -1 < index < index_first_query_word_to_appear:
                index_first_query_word_to_appear = index
        if -1 < index_first_query_word_to_appear < np.inf:
            html = html[index_first_query_word_to_appear:]

        selected_words = html.split()[:500]
        if len(selected_words) < min_n_words:
            # Don't return results with less than approx one paragraph
            continue

        html = " ".join(selected_words)

        yield {
            "href": result.get("href", ""),
            "summary": result.get("summary", ""),
            "detailed": html,
            "relevance": result.get("relevance", ""),
        }
        break

    for result in raw_results:
        yield {
            "href": result.get("href", ""),
            "summary": result.get("summary", ""),
            "relevance": result.get("relevance", ""),
        }
