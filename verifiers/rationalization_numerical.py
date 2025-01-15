from together import Together
import pandas as pd
import numpy as np
import re
import ast


class NumAutoregressiveEvaluator:
    """
    A class to evaluate hallucinations in responses of large language models for Autoregressive Trap (Numerical Responses) category.
    """

    def __init__(self,api_key, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        """
        Initialize the NumAutoregressiveEvaluator with the provided API key.

        Args:
            api_key (str): The API key for the together.ai service.
        """
        self.client = Together(api_key=api_key)
        self.model=model
        #together.api_key = api_key

    def _response_parse_llm(self, response):
        """
        Parse the response using the together.ai service.

        Args:
            response (str): The response to be parsed.

        Returns:
            str: The parsed response.
        """
        output = self.client.completions.create(
            prompt = "<Context>: 4 planets in the solar system contains the letter e. \n The 4 planets are: \n - Earth \n - Mars \n - Venus \n - Neptune \n Is there anything else I can help you with? \n\n <Instruction>: What is the numerical response and entities’ list in the above text? Just give me the number and list separated by commas \n\n <Response>: 4, earth, mars, venus, neptune \n\n <Context>: " + response + " <Instruction>: What is the numerical response and entities’ list in the above text? Just give me the number and list separated by commas \n\n",
            model=self.model,
            max_tokens = 300,
            repetition_penalty = 1.1,
            temperature = 0.1,
            stop = ["<human>:", "<Context>:", "<Instruction>:"]
        )

        return output.choices[0].text

    def _extract_atomic_units_regex(self, parsed_response):
        """
        Extract numerical and list atomic units using regular expressions.

        Args:
            parsed_response (str): The parsed response to extract atomic units from.

        Returns:
            tuple: A tuple containing the numerical atomic unit and list atomic unit.
        """
        pattern = r'(\d+),\s*((?:\s*\b\w+\b\s*,?)+)\s*'

        match = re.search(pattern, parsed_response)

        if match:
            num = int(match.group(1))
            entities_str = match.group(2).strip()
            entities = [entity.strip() for entity in entities_str.split(',')]
            return num, entities

        return None, None

    def evaluate(self, csv_file_path, output_dir="./"):
        """
        Evaluate numerical responses based on a CSV file.

        Args:
            csv_file_path (str): The path to the CSV file containing data.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        df = pd.read_csv(csv_file_path)
        #df = df.iloc[:10] #debug and test
        df['list'] = df['list'].apply(ast.literal_eval)
        df['list_condition'] = df['list_condition'].apply(ast.literal_eval)

        df['parsed_response'] = df['response'].apply(self._response_parse_llm)

        values = [{'numerical_atomic_unit': num, 'list_atomic_unit': entities} for num, entities in df['parsed_response'].apply(lambda x: self._extract_atomic_units_regex(x) or (None, None))]
        df[['numerical_atomic_unit', 'list_atomic_unit']] = pd.DataFrame(values)

        df['numerical_atomic_unit'] = df['numerical_atomic_unit'].replace(np.nan, None)

        df['true_answer'] = df.apply(lambda row: [row['count']] + row['list_condition'], axis=1)
        df['atomic_units'] = df.apply(lambda row:
                              [] if row['numerical_atomic_unit'] is None and row['list_atomic_unit'] is None
                              else row['list_atomic_unit'] if row['numerical_atomic_unit'] is None
                              else [int(row['numerical_atomic_unit'])] if row['list_atomic_unit'] is None
                              else [int(row['numerical_atomic_unit'])] + row['list_atomic_unit'],
                              axis=1)

        df['hallucinated_atomic_units'] = df.apply(lambda row: set(str(item).lower() for item in row['atomic_units']) - set(str(item).lower() for item in row['true_answer']), axis=1)

        # Construct output filename
        output_file_path = csv_file_path.split("/")[-1].replace('.csv', '_AU.csv')

        # Save the DataFrame to the output file
        df.to_csv(output_dir+output_file_path, index=False)

        # Return the filename
        return output_dir+output_file_path

if __name__ == "__main__":
    evaluator = NumAutoregressiveEvaluator(api_key="", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    filepath = evaluator.evaluate("./responses/rationalization_numerical/gpt_3.5_turbo_0125_numerical_response.csv")
    print("Output file path:")
    print(filepath)