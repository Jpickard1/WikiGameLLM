import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_page_summary(wiki_page):
    """
    Retrieves a brief summary of a given Wikipedia page.

    This function takes a Wikipedia page object and returns the summary of the page. However, rather than 
    returning the entire summary, it returns only the first few lines. This is particularly useful for 
    getting a quick overview or introduction to the page's content without needing to process the entire 
    summary text.

    Parameters
    ----------
    wiki_page : WikipediaPage object
        A Wikipedia page object from which the summary is to be extracted. The object should have a 'summary' 
        attribute containing the text of the page's summary.

    Returns
    -------
    str
        A string containing the first few lines of the Wikipedia page's summary. The exact number of lines 
        returned is set to 5 in this implementation.
    """
    # return just the first few lines if there are multiple
    return ". ".join(wiki_page.summary.split("\n")[:5])

def get_random_wiki_page(wiki_wiki):
    """
    Selects a random Wikipedia page that meets certain validity criteria.

    This function repeatedly requests random Wikipedia pages until it finds one that satisfies specific 
    criteria: the title should not start with certain prefixes (like "Template:", "List of", etc.), should 
    not contain certain unwanted characters, and must contain at least one alphabetical character. The 
    function also checks if the page has a reasonable summary (at least 20 words) before accepting it.

    Returns
    -------
    str
        The title of a valid random Wikipedia page.
    """
    wiki_title = None
    while True:
        url = "https://en.wikipedia.org/wiki/Special:Random"
        response = requests.get(url, timeout = 30, allow_redirects = True)
        final_url = response.url
        wiki_title = final_url.split("wiki/")[-1]
        is_valid_title = True

        # various unwanted prefixes
        bad_prefixes = ["list of", "history of", "Template:", "Wikipedia:", "Category:", "Portal:", "Talk:", "Template talk:"]

        # check for unwanted chars
        for char in "[]{}:%":
            if char in wiki_title:
                is_valid_title = False
        
        # validation criteria
        starts_with_bad_prefix = any(wiki_title.lower().startswith(prefix.lower()) for prefix in bad_prefixes)
        contains_alpha = any(char.isalpha() for char in wiki_title)
        is_valid_title = not starts_with_bad_prefix and contains_alpha

        if is_valid_title:

            # check if a reasonable page summary is present (at least 20 words)
            summary = get_page_summary(wiki_wiki.page(wiki_title))
            if len(summary.split()) > 20:
                break

    return wiki_title

def validate_pages(wiki_page):
    """
    Filters and validates the linked pages from a given Wikipedia page.

    This function takes a Wikipedia page object and extracts all the links (or references to other Wikipedia 
    pages) from it. It then filters out unwanted links based on predefined criteria, such as links with 
    certain prefixes (like "Template:", "Wikipedia:", etc.) and links that do not contain any alphabetical 
    characters. The purpose is to retain only relevant and potentially useful page links for further processing.

    Parameters
    ----------
    wiki_page : WikipediaPage object
        A Wikipedia page object from which the links are to be extracted and validated. The object is 
        expected to have a 'links' attribute containing a dictionary of linked page titles.

    Returns
    -------
    list
        A list of validated linked page titles. The titles in this list do not include any of the unwanted 
        prefixes and contain at least one alphabetical character.
    """
    # get all links
    links = list(wiki_page.links.keys())

    # various unwanted prefixes
    bad_prefixes = ["list of", "history of", "Template:", "Wikipedia:", "Category:", "Portal:", "Talk:", "Template talk:"]
    links = [link for link in links 
        if not any(link.lower().startswith(prefix.lower()) for prefix in bad_prefixes) 
        and any(char.isalpha() for char in link) # at least one alpha char
    ]
    return links

def get_most_similar_strings(reference_string: str, candidates_list: list[str], n = 10):
    """
    Identifies the most similar strings to a reference string from a list of candidate strings.

    This function computes the similarity between a reference string and each string in the candidate list. 
    It uses a model to generate embeddings for the reference and candidate strings, and then calculates 
    the cosine similarity between the reference embedding and each candidate embedding. The function 
    returns the top 'n' most similar strings and their similarity scores.

    Parameters
    ----------
    reference_string : str
        The reference string to which the similarity of candidate strings is to be compared.

    candidates_list : list[str]
        A list of candidate strings from which the most similar ones to the reference string are identified.

    n : int, optional
        The number of most similar strings to return. Defaults to 10.

    Returns
    -------
    tuple of (list, list)
        A tuple containing two lists: the first list contains the top 'n' most similar strings from the 
        candidates list, and the second list contains their corresponding similarity scores. The similarity 
        scores are in the range [0, 1], where 1 indicates perfect similarity.
    """
    reference_embedding = model.encode([reference_string])[0]
    encoded_strings = model.encode(candidates_list)
    topics_embs = {topic: emb for topic, emb in zip(candidates_list, encoded_strings)}
    similarities = [1 - cosine(reference_embedding, encoded_str) for encoded_str in encoded_strings]
    most_similar_indices = np.argsort(similarities)[::-1][:n]
    return topics_embs, [candidates_list[i] for i in most_similar_indices], [similarities[i] for i in most_similar_indices]

def search_wiki(search_term):
    """Search common name for search term and returns most relevant Wiki Page"""
    search_url = f"https://en.wikipedia.org/w/index.php?search={'+'.join(search_term.split())}&title=Special:Search&profile=advanced&fulltext=1&ns0=1"
    soup = BeautifulSoup(requests.get(search_url, timeout=30).content, "html.parser")
    bad_prefixes = ["list of", "history of", "Template:", "Wikipedia:", "Category:", "Portal:", "Talk:", "Template talk:"]
    for result in soup.find_all("div", class_ = "mw-search-result-heading"):
        if result.a:
            if result.a['href']:
                text = result.a['href'].replace("/wiki/", "").strip()
                starts_with_bad_prefix = any(text.lower().replace("_", " ").startswith(prefix.lower()) for prefix in bad_prefixes)
                if not starts_with_bad_prefix:
                    return text
            