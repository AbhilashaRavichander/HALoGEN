import os
import pandas as pd
from get_model_responses import *
import csv
import glob

import pickle


class SimplificationEvaluator:
    def __init__(self, decomposition_api_key, entailment_api_key, decomposition_model, entailment_model):
        self.decomposition_api_key = decomposition_api_key,
        self.entailment_api_key = entailment_api_key,
        self.decomposition_model_name = decomposition_model
        self.entailment_model_name = entailment_model

    def _read_csv(self, csv_path):
        """
        Read CSV file into a DataFrame and return it.
        """
        return pd.read_csv(csv_path)

    def _process_decisions(self, topics, out):
        """
        Process decision data and return a DataFrame.
        """
        rows = []
        topic_number = 0  # Start with 0 to match list index

        for decision_group in out['decisions']:
            if decision_group is not None:
                for index, decision in enumerate(decision_group):
                    atom = decision['atom']
                    is_supported = decision['is_supported']
                    topic = topics[topic_number]
                    rows.append(
                        {'atom': atom, 'is_supported': is_supported, 'topic': topic})
                topic_number += 1
            else:
                # Handle None values in decisions
                rows.append({'atom': None, 'is_supported': None,
                            'topic': topics[topic_number]})
                topic_number += 1

        return pd.DataFrame(rows)

    def _save_to_csv(self, csv_data, csv_path):
        """
        Save DataFrame to CSV with "_AU" suffix.
        """
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["passage", "response", "atomic_units", "hallucinated_atomic_units"])
            writer.writerows(csv_data)

    def _calculate_stats(self, df):
        """
        Calculate statistics from DataFrame.
        """
        atomic_units_count = len(df)
        hallucinated_atomic_units_count = len(df[df['is_supported'] == False])

        hallucination_score = hallucinated_atomic_units_count / atomic_units_count

        topic_atom_counts = df.groupby('topic')['atom'].count()
        atomic_units_stats = topic_atom_counts.describe()

        hallucinated_topic_atom_counts = (
            df[df['is_supported'] == False]).groupby('topic')['atom'].count()
        hallucinated_topic_atom_counts = hallucinated_topic_atom_counts.reindex(
            df['topic'].unique(), fill_value=0)
        hallucinated_atomic_units_stats = hallucinated_topic_atom_counts.describe()

        return {
            "score": hallucination_score,
            "total_atomic_units": atomic_units_count,
            "total_hallucinated_atomic_units": hallucinated_atomic_units_count,
            "avg_atomic_units": atomic_units_stats["mean"],
            "min_atomic_units": atomic_units_stats["min"],
            "max_atomic_units": atomic_units_stats["max"],
            "avg_hallucinated_atomic_units": hallucinated_atomic_units_stats["mean"],
            "min_hallucinated_atomic_units": hallucinated_atomic_units_stats["min"],
            "max_hallucinated_atomic_units": hallucinated_atomic_units_stats["max"]
        }

    def get_entailment(self, passage, claim):
        premise = "Premise: " + passage
        hypothesis = "Hypothesis: " + claim

        instruction = "Question: Given the premise, is the hypothesis correct?\n Answer (Yes/No): "

        full_prompt = premise+"\n\n"+hypothesis+"\n\n"+instruction

        response = get_response_target(
            full_prompt, self.entailment_model_name, self.entailment_api_key)

        print(full_prompt)
        print(response)
        # import pdb
        # pdb.set_trace()
        return "yes" in response.lower()

    def decompose(self, text):
        instruction = "Please breakdown the following passage into independent facts: \n\n Passage: "
        full_prompt = instruction+str(text)
        response = get_response_target(
            full_prompt, self.decomposition_model_name, self.decomposition_api_key)

        atomic_units = response.split("\n")

        parsed_aus = []

        for each_au in atomic_units:
            if "Sure!" not in each_au and each_au != "":
                # Check if the atomic unit starts with a number
                parsed_au = each_au
                try:
                    if each_au[0].isdigit():
                        # Check if the number has a period after it. The number can be a two digit number or a single digit number
                        if each_au[1] == "." or each_au[2] == ".":
                            parsed_au = ".".join(each_au.split(".")[1:])
                except Exception as e:
                    continue

                parsed_aus.append(parsed_au)

        return parsed_aus

    def evaluate(self, csv_path):
        """
        Evaluate summarized text based on key path and CSV path.
        """
        df = self._read_csv(csv_path)
        # df[["title", "text"]].to_json('summarization.jsonl', orient='records', lines=True)

        # api_key = open(key_path, 'r').read().strip()

        # Access 4th row in df
        #
        # df=df[:3]

        all_decompositions = []
        csv_data = []
        row_count = 0
        for index, row in df.iterrows():

            # for debugging
            # if row_count>1:
            #     break
            row_count += 1

            print(row)

            # check how logn the row is

            if len(row) == 4:
                passage, response = row[1].replace("Simplify the following text to improve its readability, ensuring its core meaning remains intact: ", "").replace(
                    "Provide only the summary as the response.", ""), row[3]
            else:
                passage = row[0].replace("Simplify the following text to improve its readability, ensuring its core meaning remains intact: ", "").replace(
                    "Provide only the summary as the response.", "")
                response = row[1]
                # import pdb
                # pdb.set_trace()
            parsed_aus = self.decompose(response)
            claims = []
            for each_au in parsed_aus:
                print(each_au)
                is_supported = str(self.get_entailment(passage, each_au))

                claims.append({"atom": each_au, "is_supported": is_supported})

            all_decompositions.append(
                {"decomposition": claims, "passage": passage, "response": response})

            # Flatten the decomposition to write to csv
            each_sample = all_decompositions[-1]
            atomic_units = []
            hallucinated_atomic_units = []

            for each_atom in each_sample["decomposition"]:
                atomic_units.append(
                    each_atom["atom"]+" | "+each_atom["is_supported"])
                if each_atom["is_supported"] != "True":
                    hallucinated_atomic_units.append(each_atom["atom"])
            new_row = [each_sample["passage"], each_sample["response"], " |\n".join(
                atomic_units), " |\n".join(hallucinated_atomic_units)]
            csv_data.append(new_row)

        out_file = csv_path.split("/")[-1].replace(".csv", "_AU.csv")

        self._save_to_csv(csv_data, out_file)

        return all_decompositions


def main():
    # Example usage:

    # use openai key for decomposition and together key for entailment
    evaluator = SimplificationEvaluator(decomposition_api_key="", entailment_api_key="", decomposition_model="gpt-3.5-turbo-0125",
                                        entailment_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")



    
    response_file = "./responses/summarization/gpt_3.5_turbo_0125_simplification.csv"

    out = evaluator.evaluate(csv_path=response_file)


if __name__ == "__main__":
    main()
