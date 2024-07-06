import random
import time
import string
import wikipediaapi
from funcs import *
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.memory import ConversationBufferMemory
import os
import getpass
import difflib


class WikiGameLLMBot():
    """
    A class to simulate playing the 'Wiki Game', an online game where players start from one Wikipedia article 
    and try to navigate to another target article using as few hyperlinks as possible within Wikipedia.

    This class LLMs to the Wikipedia API to fetch and analyze Wikipedia pages. It keeps track of the 
    game's progress, including the start and target topics, the current topic, and turn number.

    Attributes
    ----------
    wiki_wiki : WikipediaAPI
        The Wikipedia API connection instance.
    game_log : dict
        A dictionary to log the game's progress, including the start and target topics, turn number, 
        time taken per turn, current topic and summary, and similarity to the target.
    start_topic : str
        The starting topic of the game.
    target_topic : str
        The target topic of the game.
    current_summary : str
        The summary of the current topic.
    target_summary : str
        The summary of the target topic.

    Methods
    -------
    log_turn(turn_dict)
        Logs the details of a turn in the Wiki game.
    get_topics(start_topic=None, target_topic=None)
        Retrieves and sets the start and target topics for the Wiki game.
    take_turn(current_topic, visited)
        Processes a turn in the Wiki game, finding the most similar topic to the target on the current page.
    play_game(verbose=True, clear_cell=False)
        Starts and manages the Wiki game until the target topic is reached.

    Raises
    ------
    AssertionError
        If the start and target topics are the same, an assertion error is raised.
    """
    def __init__(self, wiki_wiki, start_topic = None, target_topic = None, model_name='meta/llama3-70b-instruct', temperature=0.1):
        """
        Initializes the WikiGameBot with the start and target topics.

        This constructor establishes a connection to the Wikipedia API, generates a unique user agent for 
        the session, and initializes the game log. It also retrieves the summaries for the start and target 
        topics, ensuring that they are not the same.

        Parameters
        ----------
        start_topic : str, optional
            The starting topic for the Wiki game. If not provided, a random Wikipedia page is chosen.
        
        target_topic : str, optional
            The target topic for the Wiki game. If not provided, a random Wikipedia page is chosen.

        Raises
        ------
        AssertionError
            If the start and target topics are the same, an assertion error is raised.
        """
        if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
            nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
            assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
            os.environ["NVIDIA_API_KEY"] = nvidia_api_key
        else:
            nvidia_api_key = os.environ["NVIDIA_API_KEY"]
            
        self.llm = ChatNVIDIA(model       = model_name,
                              api_key     = nvidia_api_key,
                              temperature = temperature,
                             )
        self.memory = ConversationBufferMemory(ai_prefix="System")

        self.wiki_wiki = wiki_wiki
        
        # game log
        self.game_log = {
            'starting_topic': [],   # to be able to calculate distance from starting topic to current topic
            'target_topic': [],     # to be able to calculate distance from target topic to current topic
            'turn': [],             # int of this turn (so things are sortable)
            'current_topic': [],    # current topic
            'current_summary': []
        }
        self.printouts = []

        # get start and target topics and their respective summaries
        self.get_topics(start_topic, target_topic) # defines self.start_topic and self.target_topic
        assert self.start_topic != self.target_topic, "Please enter different start and target topics."
        
        target_page = self.wiki_wiki.page(self.target_topic)

    def log_turn(self, turn_dict):
        """
        Logs the details of a turn in the Wiki game.

        This method updates the game log with details from the current turn, such as the turn number, time 
        taken, current topic, and similarity to the target topic.

        Parameters
        ----------
        turn_dict : dict
            A dictionary containing the details of the current turn to be logged.
        """
        for key, val in turn_dict.items():
            self.game_log[key].append(val)

    def get_topics(self, start_topic = None, target_topic = None):
        """
        Retrieves and sets the start and target topics for the Wiki game.

        This method obtains the URLs and Wikipedia page names for the start and target topics. If the topics 
        are not specified, random Wikipedia pages are chosen.

        Parameters
        ----------
        start_topic : str, optional
            The starting topic for the Wiki game.
        
        target_topic : str, optional
            The target topic for the Wiki game.
        """
        if start_topic:
            self.start_topic = search_wiki(start_topic)
        else:
            self.start_topic = get_random_wiki_page(self.wiki_wiki)
        if target_topic:
            self.target_topic = search_wiki(target_topic)
        else:
            self.target_topic = get_random_wiki_page(self.wiki_wiki)

    def take_turn(self, current_topic, visited):
        """
        Processes a turn in the Wiki game.

        This method finds the most similar topic to the target on the current page. It returns the topic 
        whose summary is most similar to the target summary.

        Parameters
        ----------
        current_topic : str
            The current topic being analyzed in the game.

        visited : list
            A list of already visited topics to avoid repetition.

        Returns
        -------
        tuple
            A tuple containing the most similar topic to the target and its similarity score.
        """
        # get API page object for topic
        current_page = self.wiki_wiki.page(current_topic)
        self.current_summary = get_page_summary(current_page) # not the best but this works

        # get top n valid pages from all linked pages
        pages = [page for page in validate_pages(current_page) if page not in visited]
        pages = pages[:25]
        if self.target_topic in pages:
            return self.target_topic, 1

        # Select the base pages with the llm
        template = """You must are playing the Wikipedia Game where you must find a chain of
Wikipedia pages that connect a source topic to a target topic. Your source topic was {source}
and your target topic is {target1}. You are currently at the topic of {current}, and you must select
a new topic from the linked pages here that will make it likely for you to connect to the target topic
soon. Your possible choices are:

{links}

Select the topic above most likely to route you to {target2} as fast as possible. Format your output as:
Next topic=<topic here>
"""
        template = template.format(source  = self.start_topic,
                                   target1 = self.target_topic,
                                   current = current_topic,
                                   links   = '\n'.join(pages),
                                   target2 = self.target_topic
                                  )
        print(template)
        response = self.llm.invoke(template)
        print("Response")
        print(response)
        
        print("Parsed Response")
        proposedPage = response.content.split('=')[1]
        print(proposedPage)
        
        print("Most similar page")
        most_similar = difflib.get_close_matches(proposedPage, pages, n=1)[0]
        print(most_similar)

        return most_similar
    
    def play_game(self, verbose = True):
        """
        Starts and manages the Wiki game until the target topic is reached.

        This method keeps track of the game, including turn number, time, and similarity scores. It outputs 
        the game progress if verbose is True. The game continues until the target topic is reached.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints detailed information about each turn. Defaults to True.

        clear_cell : bool, optional
            If True, clears the output cell every four turns (useful in interactive environments). Defaults to False.
        """
        # turn number
        turn_num = 1

        # first 'current' topic is starting topic
        current_topic = self.start_topic

        # to prevent duplicates
        visited = set()

        # keep playing until target is reached
        while True:

            # for turn time tracking
            turn_start = time.time()

            # find most similar topic on current page to target topic
            visited.add(current_topic)
            next_topic = self.take_turn(current_topic, list(visited))

            # for turn time tracking
            turn_time = time.time() - turn_start

            self.log_turn(
                {
                    'starting_topic': self.start_topic,
                    'target_topic': self.start_topic,
                    'turn': turn_num,             
                    'current_topic': current_topic,
                    'current_summary': self.current_summary
                }
            )

            if verbose:
                printouts = [
                    "-" * 50,
                    f"Turn: {turn_num}",
                    f"Start topic: {self.start_topic.replace('_', ' ')}",
                    f"Current topic: {current_topic.replace('_', ' ')}",
                    f"Next topic: {next_topic.replace('_', ' ')}",
                    f"Target topic: {self.target_topic.replace('_', ' ')}",
                ]

                self.printouts.append(printouts)

                # print progress
                for i in self.printouts[-1]:
                    print(i)

            # else, set new next_topic to current topic and loop
            current_topic = next_topic

            # increment turn
            turn_num += 1

"""
Example usage:
start_topic = "test"
target_topic = "christmas"
print(f"Starting topic: '{start_topic}'")
print(f"Target topic: '{target_topic}'")
game_bot = WikiGameBot(start_topic, target_topic)
game_bot.play_game(verbose = True, clear_cell = False)
"""