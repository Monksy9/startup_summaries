# Summarising startups from their HTML pages

**Prerequisites**

Ensure that Poetry is installed on your system. If not, please follow the official guide to install Poetry.
Navigate to your project's root directory and run the following command to install the necessary packages and set up the virtual environment:
poetry install

**Running the Application**

poetry run python app.py

**Additional Notes**

Due to the Huggingface API limitations for the free tier during testing, the application is currently configured to process a maximum of 3 companies. You can modify this limit by adjusting the relevant parameter in app.py.
A limited number of topics are used to stay within the free tier's constraints. However, you can easily expand this list by editing the .txt file located in the project's base path.
The summaries.csv file included in the repository serves as an illustrative example of company-level summaries across various topics.
