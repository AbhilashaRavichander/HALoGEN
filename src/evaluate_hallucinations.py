#!/usr/bin/env python3
"""
Model Response Evaluation Script

This script processes and evaluates model responses using various evaluators.
It supports multiple evaluation types including code analysis, summarization,
rationalization, historical events analysis, and more.

Each evaluator processes specific aspects of model responses and generates
evaluation metrics stored in the specified output folder.
"""

import os
import shutil
import yaml
from verifiers.rationalization_binary import YesNoResponseEvaluator
from verifiers.code_packages import CodeEvaluator
from verifiers.rationalization_numerical import NumAutoregressiveEvaluator
from verifiers.historical_events import InterpersonalRelationshipEvaluator
from verifiers.false_presuppositions import NumericalFalsePresuppostionEvaluator
from scientific_attribution import ReferencesEvaluator
from verifiers.summarization import SummarizationEvaluator
from verifiers.simplification import SimplificationEvaluator
from verifiers.biographies import BiographiesEvaluator
import argparse


def read_api_keys(config_file="config.yml"):
    """
    Read API keys from a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file. Defaults to "config.yml".

    Returns:
        tuple: A tuple containing (openai_api_key, together_api_key, s2_api_key)
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['openai_api_key'], config['together_api_key'], config['s2_api_key']


def get_args():
    """
    Configure and return the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Process files in a folder using various evaluators.")


    parser.add_argument("--api_key_file", default="config.yml",
                        help="Path to the text file containing API keys.")

    parser.add_argument(
        "--output_folder", help="Path to the folder where output files will be saved.", default="./")

    parser.add_argument("--model", help="Model to use from Together AI API for verifiers. Defaults to 'meta-llama/Llama-3.3-70B-Instruct-Turbo'.",
                        default="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    # Add flags for specific evaluators
    parser.add_argument("--summarization", action="store_true",
                        help="Process summarization evaluator.")
    parser.add_argument("--simplification", action="store_true",
                        help="Process summarization evaluator.")
    parser.add_argument("--code", action="store_true",
                        help="Process code evaluator.")
    parser.add_argument("--rationalization_numerical", action="store_true",
                        help="Process rationalization numerical evaluator.")
    parser.add_argument("--rationalization_binary", action="store_true",
                        help="Process rationalization binary evaluator.")
    parser.add_argument("--historical_events", action="store_true",
                        help="Process historical events evaluator.")
    parser.add_argument("--false_presupposition", action="store_true",
                        help="Process false presupposition evaluator.")
    parser.add_argument("--scientific_attribution", action="store_true",
                        help="Process scientific attribution evaluator.")
    parser.add_argument("--biographies", action="store_true",
                        help="Process biographies evaluator.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    openai_api_key, together_api_key, s2_api_key = read_api_keys(
        args.api_key_file)
    model_name_eval = "gpt_3.5_turbo_0125"

    # Initialize the processor
    # Determine selected evaluators
    selected_evaluators = []

    if args.summarization:
        response_file = "responses/summarization/"+model_name_eval + \
            "_summarization.csv"  # Model whose responses are being evaluated
        summarization_evaluator = SummarizationEvaluator(decomposition_api_key=openai_api_key, entailment_api_key=together_api_key, decomposition_model="gpt-3.5-turbo-0125",
                                                         entailment_model=args.model)
        summarization_evaluator.evaluate(
            response_file, args.output_folder)

    if args.simplification:
        response_file = "responses/simplification/"+model_name_eval + \
            "_simplification.csv"  # Model whose responses are being evaluated
        simplification_evaluator = SimplificationEvaluator(decomposition_api_key=openai_api_key, entailment_api_key=together_api_key, decomposition_model="gpt-3.5-turbo-0125",
                                                           entailment_model=args.model)
        simplification_evaluator.evaluate(
            response_file, args.output_folder)
            
    if args.code:
        response_file = "responses/code/"+model_name_eval + \
            "_code.csv"  # Model whose responses are being evaluated
        code_evaluator = CodeEvaluator()
        code_evaluator.evaluate(response_file, args.output_folder)

    if args.rationalization_numerical:
        response_file = "responses/rationalization_numerical/"+model_name_eval + \
            "_numerical_response.csv"  # Model whose responses are being evaluated
        rationalization_evaluator = NumAutoregressiveEvaluator(
            api_key=together_api_key, model_name=args.model)
        rationalization_evaluator.evaluate(
            response_file, args.output_folder)

    if args.rationalization_binary:
        response_file = "responses/rationalization_binary/"+model_name_eval + \
            "_yesno_response.csv"  # Model whose responses are being evaluated
        rationalization_evaluator = YesNoResponseEvaluator(
            api_key=together_api_key, model_name=args.model)
        rationalization_evaluator.evaluate(
            response_file, args.output_folder)

    if args.historical_events:
        response_file = "responses/historical_events/"+model_name_eval + \
            "_historicalevents.csv"  # Model whose responses are being evaluated
        historical_evaluator = InterpersonalRelationshipEvaluator(
            api_key=together_api_key, model_name=args.model)
        historical_evaluator.evaluate(
            response_file, args.output_folder)

    if args.false_presupposition:
        response_file = "responses/numerical_falsepresupposition/"+model_name_eval + \
            "_numerical_inconsistency.csv"  # Model whose responses are being evaluated
        nfp_evaluator = NumericalFalsePresuppostionEvaluator(
            api_key=together_api_key, model_name=args.model)
        nfp_evaluator.evaluate(response_file, args.output_folder)

    if args.scientific_attribution:
        pass  # You can use our public implementation (scientific_attribution.py) of the scientific attribution evaluator here but your results may be slightly different (see announcements).

    if args.biographies:
        response_file = "responses/biographies/"+model_name_eval + \
            "_biographies.csv"  # Model whose responses are being evaluated
        biograpies_evaluator = BiographiesEvaluator()
        with open('openai_key.txt', 'w') as f:
            f.write(openai_api_key)
        biograpies_evaluator.evaluate(
            "openai_key.txt", response_file, args.output_folder)
