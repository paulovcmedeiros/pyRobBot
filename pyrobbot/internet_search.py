"""Internet search module for the package."""
import ipinfo
from duckduckgo_search import DDGS

REGION = ipinfo.getHandler().getDetails().country_name


def websearch(query: str, max_results: int = 5, region: str = REGION) -> list:
    """Search the web using DuckDuckGo Search API."""
    with DDGS() as ddgs:
        yield from ddgs.text(keywords=query, region=region, max_results=max_results)
