import csv
import jsonlines
import json

def read_csv(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def write_jsonl(filename, data):
    with jsonlines.open(filename, mode='w') as writer:
        for item in data:
            writer.write(item)
