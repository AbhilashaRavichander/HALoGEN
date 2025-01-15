from together import Together
import pandas as pd
import numpy as np
import re
import ast


class NumericalFalsePresuppostionEvaluator:
    """
    A class to evaluate hallucinations in responses of large language models for numerical false presuppositions category.
    """

    def __init__(self,api_key, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        """
        Initializes the NumericalFalsePresuppositionEvaluator with the provided API key and default model.

        Args:
            api_key (str): The API key for the together.ai service.
            model (str): The default model name to use. Defaults to "meta-llama/Llama-3.3-70B-Instruct-Turbo".

        """
        self.client = Together(api_key=api_key)
        self.model = model


    def _response_parse_llm(self, response):
        """
        Parses the response using the together.ai service.

        Args:
            response (str): The response to be parsed.

        Returns:
            str: The parsed response.
        """
        output = self.client.completions.create(
            prompt = "<Context>: Sure, here is the list of plantes 1. Mercury; 2. Mars; 3. Mimas; 4. Mab; 5. Miranda; 6. Metis; 7. Melpomene.```### List with `and```` \n \n <Instruction>: What is the entities’ list in the above text? Just give the list separated by commas \n\n <Response>: Mercury, Mars, Mimas, Mab, Miranda, Metis, Melpomene \n\n <Context>: " + response + " <Instruction>: What is the entities’ list in the above text? Just give the list separated by commas \n\n <Response>:",
            model=self.model,
            max_tokens = 300,
            repetition_penalty = 1.1,
            stop = ["<human>:", "<Context>:", "<Instruction>:"]
        )

        return output.choices[0].text

    def _extract_atomic_units(self, text):
        """
        Extracts atomic units from the given text.

        Args:
            text (str): The text to extract atomic units from.

        Returns:
            list: A list of atomic units.
        """
        if ', ' in text:
            atom = text.split(', ')
            return atom
        else:
            return []

    def evaluate(self, csv_file_path, output_dir="./"):
        """
        Evaluates numerical false presuppositions based on a CSV file.

        Args:
            csv_file_path (str): The path to the CSV file containing data.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        df = pd.read_csv(csv_file_path)
        #df = df.sample(n=10, replace=False) #debug and test
        df['list'] = df['list'].apply(ast.literal_eval)
        df['list_condition'] = df['list_condition'].apply(ast.literal_eval)

        df['parsed_response'] = df['response'].apply(self._response_parse_llm)

        df['atomic_units'] = df['parsed_response'].apply(self._extract_atomic_units)

        df['hallucinated_atomic_units'] = df.apply(lambda row: list(set(str(item).lower().strip() for item in row['atomic_units']) - set(str(item).lower().strip() for item in row['list_condition'])), axis=1)

        # Update 'atomic_units' and 'hallucinated_atomic_units' columns by handindling "no response"
        condition = df['response'].str.lower().str.contains("no response", na=False)
        df.loc[condition, 'atomic_units'] = df.loc[condition, 'atomic_units'].apply(lambda x: [])
        df.loc[condition, 'hallucinated_atomic_units'] = df.loc[condition, 'hallucinated_atomic_units'].apply(lambda x: [])
        
        # Construct output filename
        output_file_path = csv_file_path.split("/")[-1].replace('.csv', '_AU.csv')

        # Save the DataFrame to the output file
        df.to_csv(output_dir+output_file_path, index=False)

        # Return the filename
        return output_dir+output_file_path


if __name__ == "__main__":
    evaluator = NumericalFalsePresuppostionEvaluator(api_key="", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    filepath = evaluator.evaluate("./responses/numerical_falsepresuppositions/gpt_3.5_turbo_0125_numerical_inconsistency.csv")
    print("Output file path:")
    print(filepath)
