"""
This file is the framework for generating multiple Streamlit applications
through an object oriented framework.

Adapted from:
<https://towardsdatascience.com/creating-multipage-applications-using-streamlit-efficiently-b58a58134030>
"""

# Import necessary libraries
import streamlit as st


# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = {}
        # Keep track of which page we're on, so we remain in it when adding a new page
        self.selected_chat_index = None

    def add_page(self, page_id, title, func) -> None:
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages[page_id] = {"title": title, "function": func}
        # Signal to `run` taht we should move to the newly added page
        self.selected_chat_index = list(self.pages.keys()).index(page_id)

    def run(self):
        # Drodown to select the page to run
        if id_and_page := st.sidebar.selectbox(
            label="Select Chat",
            options=self.pages.items(),
            format_func=lambda id_and_page: id_and_page[1]["title"],
            index=self.selected_chat_index,
        ):
            # run the app function
            page_id, page = id_and_page
            page["function"](page_id=page_id)
