import duckduckgo_search

from pyrobbot.internet_utils import websearch

# if called inside tests or fixtures. Leave it like this for now.
try:
    search_results = list(websearch("foobar"))
except duckduckgo_search.exceptions.RateLimitException:
    search_results = []


def test_websearch():
    for i_result, result in enumerate(search_results):
        assert isinstance(result, dict)
        assert ("detailed" in result) == (i_result == 0)
        for key in ["summary", "relevance", "href"]:
            assert key in result
