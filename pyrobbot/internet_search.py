"""Internet search module for the package."""
from duckduckgo_search import DDGS

from . import GeneralConstants


def websearch(
    query: str,
    max_results: int = 5,
    region: str = GeneralConstants.IPINFO["country_name"],
) -> list:
    """Search the web using DuckDuckGo Search API."""
    with DDGS() as ddgs:
        for result in ddgs.text(keywords=query, region=region, max_results=max_results):
            yield {"href": result["href"], "body": result["body"]}
