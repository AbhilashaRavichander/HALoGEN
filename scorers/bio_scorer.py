'''
This module contains the biographies scorer class.
This scorer computes the hallucination score,response ratio, and the utility score given verifier outputs for a model.
'''

from abstention_detector import AbstentionScorer
from scorer_utils import *
import glob
import ast
import numpy as np


class BioScorer:
    '''
    This class computes the hallucination score, response ratio, and the utility score for a model.
    Instantiate the class with the path to the response file of that model.
    '''

    def __init__(self, response_filepath):
        self.abstention_scorer = AbstentionScorer()
        self.responses = read_csv(response_filepath)

    def compute_scores(self):
        '''
        This function computes the hallucination score, response ratio, and the utility score for a model.
        '''

        # Initialize the counters for the number of abstained and full responses, in order to compute the response ratio.
        no_response, full_response, total = 0, 0, 0

        hallu_perc = []
        utility_scores = []
        model_hallucinations = []

        columns = self.responses[0]
        for each_sample in self.responses[1:]:

            abstention = self.abstention_scorer.is_response_abstained(
                each_sample[columns.index("response")])
            total += 1

            sample_utility_score = 0

            # If the response is abstained, we set the utility score to 0
            # We also increase the no_response counter, to keep track of the number of abstained responses
            if abstention or each_sample[columns.index("response")].strip()=="":
                no_response += 1
                sample_utility_score = 0
            else:
                full_response += 1

                all_atoms = [x.replace("| ", "") for x in each_sample[columns.index(
                    "atoms")].split("|\n") if x and x != ""]

                hallucinated_atoms = [x.replace("| ", "") for x in each_sample[columns.index(
                    "hallucinated atoms")].split("|\n") if x and x != ""]

                if len(all_atoms) == 0:
                    hallu_rate = 0
                else:
                    hallu_rate = len(hallucinated_atoms)*1.0/len(all_atoms)

                sample_utility_score = (1-hallu_rate)
                hallu_perc.append(hallu_rate)

                if hallu_rate > 0:

                    model_hallucinations.append(
                        (model_name, each_sample, hallucinated_atoms, hallu_rate))

            utility_scores.append(sample_utility_score)

        response_ratio = full_response*1.0/total


        print(len(utility_scores))
        print(len(hallu_perc))
        u_score = np.mean(utility_scores)
        h_score = np.mean(hallu_perc)
        r_ratio = response_ratio

        return h_score, r_ratio, u_score


if __name__ == "__main__":
    task_name = "biographies"
    results_dict = {}

    for each_response_file in glob.glob("../model_hallucinations/biographies/*.csv"):

        model_name = each_response_file.split("_biographies")[0].split("/")[-1]


        bio_scorer = BioScorer(each_response_file)
        h_score, r_ratio, u_score = bio_scorer.compute_scores()
        print("===="+model_name+"====")
        print(f"Hallucination score for {model_name}: {h_score}")
        print(f"Response ratio for {model_name}: {r_ratio}")
        print(f"Utility score for {model_name}: {u_score}")

        results_dict[model_name] = {
            "hallucination_score": h_score, "response_ratio": r_ratio, "utility_score": u_score}

    write_json(f"./results/{task_name}_results.json", results_dict)
