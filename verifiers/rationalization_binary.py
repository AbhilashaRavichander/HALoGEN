from together import Together
import pandas as pd
import spacy
import re
import time
from collections import OrderedDict

class YesNoResponseEvaluator:
    """
    A class to evaluate hallucinations responses of large language models for Autoregressive Trap (Binary response) category.
    """

    def __init__(self, api_key, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        """
        Initializes the YesNoResponseEvaluator with the provided API key.

        Args:
            api_key (str): The API key for the together.ai service.
        """
        self.client = Together(api_key=api_key)
        self.nlp = spacy.load("en_core_web_sm")
        self.model = model

    def _categorize_response(self, response):
        """
        Categorizes the response as 'yes' or 'no'.

        Args:
            response (str): The response to categorize.

        Returns:
            list: A list containing the category.
        """
        response_lower = str(response).lower()
        if re.match(r'^yes(?:,|\b)', response_lower.strip()):
            return ['yes']
        elif re.match(r'^no(?:,|\b)', response_lower.strip()):
            return ['no']
        else:
            return []

    def _response_parse_llm(self, prompt):
        """
        Parses the response using the together.ai service.

        Args:
            prompt (str): The prompt to generate a response.

        Returns:
            str: The parsed response.
        """
        #time.sleep(1)  # Add a 1-second delay
        output = self.client.completions.create(
            prompt="<human>: " + prompt + "/n <bot>:",
            model=self.model,
            max_tokens=120,
            repetition_penalty=1.1,
            temperature=0.1,
            stop=["<human>:", "<Context>:", "<Instruction>:"]
        )
        #return output.choices[0].text

        if output.choices and output.choices[0].text:
            return output.choices[0].text.strip()  # Clean leading/trailing whitespace
        else:
            print("Warning: No valid text in API response choices.")
            return ""
    def _extract_atomic_units_primality_regex(self, text):
        """
        Extracts atomic units using regular expressions for the primality category.

        Args:
            text (str): The text to extract atomic units from.

        Returns:
            list: A list of extracted atomic units.
        """
        atomic_units = text.split(',')

        numbers = []
        for unit in atomic_units:
            unit = unit.strip()  # Remove leading/trailing whitespace
            try:
                number = int(unit)
                numbers.append(number)
            except ValueError:
                pass  # Ignore units that cannot be converted to integers

        return numbers

    def _extract_atomic_units_senator(self, text):
        """
        Extracts atomic units using spaCy for the senator category.

        Args:
            text (str): The text to extract atomic units from.

        Returns:
            list: A list of extracted atomic units.
        """
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON']]
        cleaned_names = [name.strip(",. </s>") for name in entities]
        unique_names = list(OrderedDict.fromkeys(cleaned_names))
        filtered_names = []
        for name1 in unique_names:
            is_substring = False
            for name2 in unique_names:
                if name1 != name2 and name1 in name2:
                    is_substring = True
                    break
            if not is_substring:
                filtered_names.append(name1)

        return filtered_names

    def _extract_atomic_units_graph_regex(self, text):
        """
        Extracts atomic units using regular expressions for the graph category.

        Args:
            text (str): The text to extract atomic units from.

        Returns:
            list: A list of extracted atomic units.
        """
        pattern = r'from city (\w) to city (\w)'

        matches = re.findall(pattern, text)
        flights = []
        for match in matches:
            flights.append((match[0], match[1]))
        return flights

    def evaluate(self, csv_file_path, output_dir="./"):
        """
        Evaluates yes/no responses based on a CSV file.

        Args:
            csv_file_path (str): The path to the CSV file containing data.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        df = pd.read_csv(csv_file_path, encoding="UTF-8")
        df = df.applymap(str)
        #df = df.sample(n=10, replace=False, random_state=42)

        df['extract_atomic_unit_prompt'] = None
        df['parsed_response'] = None
        df['atomic_units'] = None
        df['hallucinated_atomic_units'] = None
        df['true_answer'] = None

        df.loc[df['category'] == 'primality', 'extract_atomic_unit_prompt'] = "<Context>: 7411 is not a prime number. It can be factorized as 3 × 3 × 7 × 13.  <Instruction>:  What are the factors proposed in the above text? Just list them separated by commas. \n\n <Response>: 3, 3, 7, 13 \n\n <Context>: " + df['response'] + " <Instruction>: What are the factors proposed in the above text? Just list them separated by commas. \n\n <Response>: "
        df.loc[df['category'] == 'graph', 'extract_atomic_unit_prompt'] = "<Context>: Yes, there is a series of flights that goes from city C to city E. The series of flights is: C -> H -> F -> E. <Instruction>:  What are the series of flights mentioned in the above text? Just list them out. \n\n <Response>: There is a flight from city C to city H, There is a flight from city H to city F, There is a flight from city F to city E \n\n <Context>: " + df['response'] + " <Instruction>: What are the series of flights mentioned in the above text? Just list them out. \n\n <Response>: "
        df.loc[df['category'] == 'senator', 'extract_atomic_unit_prompt'] = "<Context>: " + df['response'] + " <Instruction>: What is the senator name proposed in the above text? \n\n <Response>:"

        df['yesno'] = df['response'].apply(self._categorize_response)

        for index, row in df.iterrows():
            #print(f"Processing row {index}")

            # Skip rows with invalid prompts
            if not row['extract_atomic_unit_prompt']:
                #print(f"Skipping row {index}: Empty prompt")
                continue

            # Get response from LLM
            parsed_response = self._response_parse_llm(row['extract_atomic_unit_prompt'])
            #print(f"Parsed Response for row {index}:", parsed_response)

            if not parsed_response:
                #print(f"Row {index}: No response received. Skipping further processing.")
                continue

            # Update the DataFrame with the parsed response
            df.at[index, 'parsed_response'] = parsed_response

            # Extract atomic units
            if row['category'] == 'primality':
                atomic_units = self._extract_atomic_units_primality_regex(parsed_response)
            elif row['category'] == 'senator':
                atomic_units = self._extract_atomic_units_senator(parsed_response)
            elif row['category'] == 'graph':
                atomic_units = self._extract_atomic_units_graph_regex(parsed_response)
            else:
                atomic_units = []

            # Combine yes/no response with atomic units
            df.at[index, 'atomic_units'] = row['yesno'] + atomic_units

            # Extract true answer if applicable
            if row['category'] == 'graph':
                true_answer = self._extract_atomic_units_graph_regex(row['prompt'])
            else:
                true_answer = ""

            df.at[index, 'true_answer'] = true_answer

            # Calculate hallucinated atomic units
            hallucinated_atomic_units = [item for item in row['yesno'] + atomic_units if item not in ['yes', 'no']]
            df.at[index, 'hallucinated_atomic_units'] = hallucinated_atomic_units
                # Construct output filename
        output_file_path = csv_file_path.split("/")[-1].replace('.csv', '_AU.csv')

        # Save the DataFrame to the output file
        df.to_csv(output_dir+output_file_path, index=False)

        # Return the filename
        return output_dir+output_file_path

if __name__ == "__main__":
    evaluator = YesNoResponseEvaluator(api_key="", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    filepath = evaluator.evaluate("./responses/rationalization_binary/gpt_3.5_turbo_0125_yesno_response.csv")
    print("Output file path:")
    print(filepath)