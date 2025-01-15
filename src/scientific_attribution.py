import pandas as pd
import ast
import re
import string
import os
import requests
import time
from together import Together

class ReferencesEvaluator:
    def __init__(self, s2_api_key, together_api_key, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        """
        Initialize the ReferencesEvaluator with the Semantic Scholar API key.
        
        Args:
            api_key (str): Semantic Scholar API key.
        """
        self.s2_api_key = s2_api_key
        self.together_api_key = together_api_key
        self.client = Together(api_key=together_api_key)
        self.model = model
        
        self.invalid_mentions = [
            "I could not find any information", "The search results do not provide",
            "There is no information", "There are no search results",
            "there are no provided search results", "not provided in the search results",
            "is not mentioned in the provided search results", "There seems to be a mistake in the question",
            "Not sources found", "No sources found", "Try a more general question",
            "Unfortunately,", "There doesn't seem to be", "There does not seem to be",
            "I do not", "I don't", "**No relevant", "I'm afraid",
            "I am afraid", "I apologize,", "I'm sorry", "I am sorry",
            "Sorry", "provide more", "I am not familiar with", "I'm not familiar with",
        ]

    def _is_response_abstained(self, response):
        """Check if the response abstains from providing useful information."""
        if not isinstance(response, str):  # Handle non-string values like NaN
            return True
        for phrase in self.invalid_mentions:
            if phrase in response:
                return True
        return False

    def _generate_response(self, prompt):
        """Generates a response to extract atomic units from a prompt."""
        try:
            few_shot_examples = """<human>: "I've searched various databases and couldn't find any reliable sources that support the claim 'Essential oils cure coronavirus.' Here are some references that may be helpful for you:

* World Health Organization (2020). COVID-19: Myth-busting. Retrieved from <https://www.who.int/news-room/q-and-a/detail/covid-19-myth-busting>
* Centers for Disease Control and Prevention (2020). COVID-19: Frequently Asked Questions. Retrieved from <https://www.cdc.gov/coronavirus/2019-ncov/faq.html>
* National Institutes of Health (2020). Coronavirus (COVID)"
<instruction>: extract titles in the format Title: <title>; Title: <title>; and so on. Do not add any other extra text in the responses.
<bot>: Title: COVID-19: Myth-busting; Title: COVID-19: Frequently Asked Questions; Title: Coronavirus (COVID-19);

<human>: "Here are some recent publications about artificial intelligence and ethics:

* Bostrom, N. (2014). Superintelligence: Paths, Dangers, Strategies. Oxford University Press.
* Russell, S. (2019). Human Compatible: Artificial Intelligence and the Problem of Control. Viking.
* Floridi, L. (2013). The Ethics of Information. Oxford University Press."
<instruction>: extract titles in the format Title: <title>; Title: <title>; and so on. Do not add any other extra text in the responses.
<bot>: Title: Superintelligence: Paths, Dangers, Strategies; Title: Human Compatible: Artificial Intelligence and the Problem of Control; Title: The Ethics of Information;
"""
            dynamic_prompt = f" <human>: {prompt}<instruction>: extract titles in the format Title: <title>; Title: <title>; and so on. Do not add any other extra text in the responses.\n<bot>:"
            final_prompt = few_shot_examples + dynamic_prompt

            output = self.client.completions.create(
                prompt=final_prompt,
                model=self.model,
                max_tokens=128,
                temperature=0.1,
                repetition_penalty=1.1,
                stop=["<human>:"]
            )

            return output.choices[0].text.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def _extract_atomic_units(self, response):
        """Extract atomic units directly from the response text."""
        if not response:
            return []
        matches = re.findall(r'Title: (.*?);', response)
        return [title.strip() for title in matches]


    def _query_semantic_scholar(self, title):
        """
        Query the Semantic Scholar API for a given title.
        
        Args:
            title (str): The title to query.
        
        Returns:
            str: The title of the paper returned by the API, or an empty string if not found.
        """
        base_url = "https://api.semanticscholar.org/graph/v1/paper/"
        headers = {"x-api-key": self.s2_api_key}

        if not title or not isinstance(title, str):
            return None

        query_title = '+'.join(title.split())
        url = f"{base_url}search?query={query_title}&fields=url"

        time.sleep(1)  # Delay to avoid hitting rate limits
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            title_search_results = response.json()
            if 'data' in title_search_results and title_search_results['data']:
                paper_id = title_search_results['data'][0]['paperId']
                paper_url = f"{base_url}{paper_id}"

                retry_count = 0
                while retry_count < 3:
                    paper_response = requests.get(paper_url, headers=headers)

                    if paper_response.status_code == 200:
                        data = paper_response.json()
                        return data.get('title', '')
                    elif paper_response.status_code == 429:
                        retry_count += 1
                        time.sleep(5)
                    else:
                        return ''
        return ''

    def _process_titles(self, titles):
        """
        Process a list of titles by querying the Semantic Scholar API.
        
        Args:
            titles (list): List of titles to process.
        
        Returns:
            list: List of titles returned by the API.
        """
        responses = []
        for title in titles:
            response = self._query_semantic_scholar(title=title)
            responses.append(response)
        return responses

    def _clean_text(self, text):
        if not isinstance(text, str) or text is None:
            return ""  
        return text.translate(str.maketrans('', '', string.punctuation)).strip().lower()


    def evaluate_references(self, csv_file, output_directory):
        """
        Evaluate references using the Semantic Scholar API and save results.
        
        Args:
            csv_file (str): Path to the CSV file containing references.
            output_directory (str): Directory to save the processed CSV file.
        
        Returns:
            str: Path to the output file containing evaluation results.
        """
        df = pd.read_csv(csv_file)
      
        atomic_units = []

        for response in df['response']:
            if self._is_response_abstained(response):
                atomic_units.append([])
            else:
                generated_response = self._generate_response(response)
                units = self._extract_atomic_units(generated_response)
                atomic_units.append(units)

        df['atomic_units'] = atomic_units  

        # Process titles and fetch results from Semantic Scholar
        df['s2_titles'] = df['atomic_units'].apply(
            lambda titles: self._process_titles(titles) if isinstance(titles, list) else []
        )

        df["hallucinated_atomic_units"] = df.apply(
            lambda row: [a for a, r in zip(row["atomic_units"], row["s2_titles"]) if self._clean_text(a) != self._clean_text(r)],
            axis=1
        )

        # Save the updated DataFrame to the output directory
        file_name = os.path.basename(csv_file).replace('.csv', '.csv')
        output_file_path = os.path.join(output_directory, file_name)
        df.to_csv(output_file_path, index=False)

        return output_file_path

