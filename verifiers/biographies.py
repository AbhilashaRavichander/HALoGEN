import os
import pandas as pd
from factscore.factscorer import FactScorer
import nltk

class BiographiesEvaluator:
    def __init__(self):
        nltk.download('punkt')

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
                for decision in decision_group:
                    atom = decision['atom']
                    is_supported = decision['is_supported']
                    topic = topics[topic_number]
                    support_status = "True" if is_supported else "False"
                    rows.append({'atom': f"| {atom}\t{support_status} |\n", 'is_supported': is_supported, 'topic': topic})
                topic_number += 1
            else:
                # Handle None values in decisions
                rows.append({'atom': "| None\tNone |\n", 'is_supported': None, 'topic': topics[topic_number]})
                topic_number += 1

        return pd.DataFrame(rows)

    def evaluate(self, key_path, csv_path, output_dir="./"):
        """
        Evaluate biographies based on key path and CSV path.
        """
        df = self._read_csv(csv_path)
        #df = df.head(2)
        topics = df['name'].tolist()
        generations = df['response'].tolist()

        fact_scorer = FactScorer(openai_key=key_path)
        out = fact_scorer.get_score(topics, generations, gamma=10)

        df_decisions = self._process_decisions(topics, out)

        grouped_df = df_decisions.groupby('topic').agg(
            atoms=('atom', lambda x: "".join(x)),
            hallucinated_atoms=('atom', lambda x: "".join(atom for atom in x if "False" in atom))
        ).reset_index()

        # Ensure the output matches the original file's structure
        output_df = pd.DataFrame({
            'name': grouped_df['topic'],
            'response': grouped_df['topic'].map(df.set_index('name')['response']),
            'atoms': grouped_df['atoms'],
            'hallucinated atoms': grouped_df['hallucinated_atoms']
        })

        output_file_name = os.path.basename(csv_path).replace('.csv', '.csv')
        output_file_path = os.path.join(output_dir, output_file_name)
        
        output_df.to_csv(output_file_path, index=False)
        
        return output_file_path

if __name__ == "__main__":
    evaluator = BiographiesEvaluator()

    # Example usage
    filepath = evaluator.evaluate(key_path="", csv_path="./responses/biographies/gpt_3.5_turbo_0125_biographies.csv")
    print("Output file path:")
    print(filepath)

