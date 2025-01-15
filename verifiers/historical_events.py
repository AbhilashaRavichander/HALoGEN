from together import Together
import pandas as pd

class InterpersonalRelationshipEvaluator:
    """
    A class to evaluate hallucinations in responses of large language models for Interpersonal Relationship category.
    """

    def __init__(self,api_key, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        """
        Initialize the InterpersonalRelationshipEvaluator with the provided API key.

        Args:
            api_key (str): The API key for the together.ai service.
        """
        #together.api_key = api_key
        self.client = Together(api_key=api_key)
        self.model=model

    def _response_parse_llm(self, prompt):
        """
        Parse the response using the together.ai service.

        Args:
            prompt (str): The prompt for parsing the response.

        Returns:
            str: The parsed response.
        """
        output = self.client.completions.create(
            prompt = "<human>: " + prompt + "/n <bot>:",
            model= self.model,
            temperature = 0.1,
            max_tokens = 20,
            repetition_penalty = 1.1,
            stop = ["<human>:", "<Context>:", "<Instruction>:"]
          )

        return output.choices[0].text

    def evaluate(self, csv_file_path, output_dir="./"):
        """
        Evaluate interpersonal relationships based on a CSV file containing verification text.

        Args:
            csv_file_path (str): The path to the CSV file containing data.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        df = pd.read_csv(csv_file_path)
        #df = df.iloc[:10]
        df['responds'] = df.apply(lambda row: 'meeting' in row['response'].lower() or
                                      any(part.lower() in row['response'].lower() for part in row['name1'].split()) or
                                      any(part.lower() in row['response'].lower() for part in row['name2'].split()), axis=1)
        df['verification'] = ' <Context>: ' + df['response'] + ' \n <Instruction>: Does above text comfirm the occurrence of meeting between ' + df['name1'] + ' and ' + df['name2'] + '? Answer in just yes or no. <Response>: '
        df['verified_response'] = df['verification'].apply(self._response_parse_llm)
        #df['atomic_units'] = df['verified_response'].apply(lambda response: ['yes'] if 'yes' in response.lower() else ['no'] if 'no' in response.lower() else [])
        #df['hallucinated_atomic_units'] = df['atomic_units'].apply(lambda response: ['yes'] if response == ['yes'] else [])
        df['atomic_units'] = df.apply(lambda row: ['yes'] if row['responds'] and 'yes' in row['verified_response'].lower() else ['no'] if row['responds'] and 'no' in row['verified_response'].lower() else [], axis=1)
        df['hallucinated_atomic_units'] = df.apply(lambda row: ['yes'] if row['responds'] and row['atomic_units'] == ['yes'] else [], axis=1)

        # Construct output filename
        output_file_path = csv_file_path.split("/")[-1].replace('.csv', '_AU.csv')

        # Save the DataFrame to the output file
        df.to_csv(output_dir+output_file_path, index=False)

        # Return the filename
        return output_dir+output_file_path


if __name__ == "__main__":
    evaluator = InterpersonalRelationshipEvaluator(api_key="", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    filepath = evaluator.evaluate("./responses/historical_events/gpt_3.5_turbo_0125_historical_events.csv")
    print("Output file path:")
    print(filepath)
