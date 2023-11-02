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

    def add_page(self, page_id, title, func) -> None:
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages[page_id] = {"title": title, "function": func}

    def run(self):
        # Drodown to select the page to run
        id_and_page = st.sidebar.selectbox(
            label="Select Chat",
            options=self.pages.items(),
            format_func=lambda id_and_page: id_and_page[1]["title"],
        )

        # run the app function
        if id_and_page is not None:
            page_id, page = id_and_page
            page["function"](page_id=page_id)
