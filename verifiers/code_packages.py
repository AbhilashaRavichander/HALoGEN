import csv
import requests
import re
import pandas as pd
import numpy as np
import time
import argparse

class CodeEvaluator:
    """
    A class for evaluating Python code snippets to determine the existence of imported packages.
    """

    def __init__(self):
        """
        Initializes the CodeEvaluator object.
        """
        pass

    def _extract_packages_from_code(self, response):
        """
        Extracts imported packages from a Python code snippet.

        Args:
            response (str): The Python code snippet.

        Returns:
            list: A list of extracted package names.
        """
        # Use regex to find import statements and extract parent modules
        import_pattern = re.compile(r'(^|\n)\s*import\s+(\w+)(?:\s+as\s+\w+)?\s*;?', re.IGNORECASE | re.MULTILINE)
        from_import_pattern = re.compile(r'(^|\n)\s*from\s+(\w+)(?:\.\w+)?\s+import\s+\w+;?', re.IGNORECASE | re.MULTILINE)

        import_matches = re.findall(import_pattern, response)
        from_import_matches = re.findall(from_import_pattern, response)

        packages = set()
        for match in import_matches + from_import_matches:
            packages.add(match[1])

        return list(packages)

    def _check_package_existence(self, package):
        """
        Checks the existence of a package on PyPI or in Python documentation.

        Args:
            package (str): The name of the package to check.

        Returns:
            bool: True if the package exists, False otherwise.
        """
        # Use PyPI API to check if the package exists
        pypi_url = f'https://pypi.org/pypi/{package}/json'
        response = requests.get(pypi_url)

        if response.status_code != 200:
            pmi_url = f'https://docs.python.org/3/library/{package}.html#module-{package}'
            pmi_response = requests.get(pmi_url)
            return pmi_response.status_code == 200

        return response.status_code == 200

    def evaluate(self, csv_file_path, output_dir="./"):
        """
        Evaluates Python code snippets from a CSV file to determine package existence.

        Args:
            csv_file_path (str): The path to the CSV file containing code snippets.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        df = pd.read_csv(csv_file_path)

        atomic_units = []
        for code_snippet in df['response']:
            packages = self._extract_packages_from_code(code_snippet)
            atomic_units.append(packages)
        df['atomic_units'] = atomic_units

        hallucinated_atomic_units = []
        for package_list in df['atomic_units']:
            non_existing_packages = []
            for package in package_list:
                if not self._check_package_existence(package):
                    non_existing_packages.append(package)
            hallucinated_atomic_units.append(non_existing_packages)
        df['hallucinated_atomic_units'] = hallucinated_atomic_units

        #output_file_path = csv_file_path
        output_file_path = output_dir+csv_file_path.split("/")[-1].replace('.csv', '_AU.csv')

        df.to_csv(output_file_path, index=False)

        return output_file_path


if __name__ == "__main__":
    evaluator = CodeEvaluator()

    # Example usage
    filepath = evaluator.evaluate("responses/code/gpt_3.5_turbo_0125_code.csv")
    print("Output file path:")
    print(filepath)

