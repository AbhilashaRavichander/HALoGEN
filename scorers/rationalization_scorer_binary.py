'''
This module contains the scorer class for binary rationalization tasks.
This scorer computes the hallucination score,response ratio, and the utility score given verifier outputs for a model.
'''

from abstention_detector import AbstentionScorer
from scorer_utils import *
import glob
import ast
import numpy as np
import nltk
import ast


def write_csv(data, filepath):
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def write_jsonl_to_csv(data, filepath):
    key_names = [
        "prompt",
        "response",
        "atomic_units",
        "hallucinated_atomic_units",
        "abstention",
        "hallu_rate",
        "sample_utility_score"]
    full_data = [key_names]

    for each_sample in data:
        full_data.append([each_sample["prompt"], each_sample["response"], each_sample["atomic_units"], each_sample["hallucinated_atomic_units"], str(
            each_sample["abstention"]), str(each_sample["hallu_rate"]), str(each_sample["sample_utility_score"])])

    write_csv(full_data, filepath)


class RationalizationScorerBinary:
    '''
    This class computes the hallucination score, response ratio, and the utility score for a model.
    Instantiate the class with the path to the response file of that model.
    '''

    def __init__(self, response_filepath):
        self.abstention_scorer = AbstentionScorer()
        self.responses = read_csv(response_filepath)
        self.response_filepath = response_filepath

    def compute_scores(self):
        '''
        This function computes the hallucination score, response ratio, and the utility score for a model.
        '''

        # Initialize the counters for the number of abstained and full responses, in order to compute the response ratio.
        no_response, full_response, total = 0, 0, 0

        hallu_perc = []
        utility_scores = []
        model_hallucinations = []

        model_name = self.response_filepath.split("_yesno")[0].split("/")[-1]

        category_to_correct_response = {
            "primality": "yes", "senator": "no", "graph": "no"}

        full_data = []

        columns = self.responses[0]
        for each_sample in self.responses[1:]:

            abstention = self.abstention_scorer.is_response_abstained(
                each_sample[columns.index("response")])
            total += 1

            sample_utility_score = 0

            # If the response is abstained, we set the utility score to 0
            # We also increase the no_response counter, to keep track of the number of abstained responses
            if abstention or each_sample[columns.index("response")].strip() == "":
                no_response += 1
                sample_utility_score = 0
            else:
                full_response += 1

                category = each_sample[columns.index("category")]
                response = each_sample[columns.index("response")]

                all_atoms = ast.literal_eval(
                    each_sample[columns.index("atomic_units")])
                hallucinated_atoms = ast.literal_eval(
                    each_sample[columns.index("hallucinated_atomic_units")])

                # We take the first line of the response since we ask the model for its answer first
                response_words = [x.lower() for x in nltk.word_tokenize(
                    response.split("\n")[0])]

                # Either there was no yes, no, or factorization in answer (ie no proposed atomic facts), or the answer had the correct response
                if category_to_correct_response[category].lower() in response_words or len(all_atoms) == 0:
                    hallu_rate = 0
                else:
                    hallu_rate = len(hallucinated_atoms)*1.0/len(all_atoms)

                sample_utility_score = (1-hallu_rate)

                hallu_perc.append(hallu_rate)

            utility_scores.append(sample_utility_score)

        response_ratio = full_response*1.0/total

        print(len(utility_scores))
        print(len(hallu_perc))
        u_score = np.mean(utility_scores)
        h_score = np.mean(hallu_perc)
        r_ratio = response_ratio

        return h_score, r_ratio, u_score


if __name__ == "__main__":
    task_name = "rationalization_binary"
    results_dict = {}

    for each_response_file in glob.glob("../model_hallucinations/rationalization_binary/*.csv"):

        model_name = each_response_file.split("_yesno")[0].split("/")[-1]

        bin_scorer = RationalizationScorerBinary(each_response_file)
        h_score, r_ratio, u_score = bin_scorer.compute_scores()
        print("===="+model_name+"====")
        print(f"Hallucination score for {model_name}: {h_score}")
        print(f"Response ratio for {model_name}: {r_ratio}")
        print(f"Utility score for {model_name}: {u_score}")

        results_dict[model_name] = {
            "hallucination_score": h_score, "response_ratio": r_ratio, "utility_score": u_score}

    write_json(f"./results/{task_name}_results.json", results_dict)
