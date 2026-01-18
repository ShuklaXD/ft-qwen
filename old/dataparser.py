import csv
import json

def convert_csv_to_jsonl(input_csv_path, output_jsonl_path):
    """
    Converts a CSV file with 'question' and 'answer' columns
    into a JSONL file suitable for fine-tuning.
    """
    with open(input_csv_path, 'r', encoding='utf-8') as csv_file, \
         open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            question = row.get('question')
            answer = row.get('answer')

            if question is None or answer is None:
                print(f"Skipping row due to missing 'question' or 'answer': {row}")
                continue

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            jsonl_file.write(json.dumps({"messages": messages}) + '\n')

if __name__ == "__main__":
    # Example usage:
    # Create a dummy CSV file for demonstration if it doesn't exist
    try:
        with open("train.csv", 'x', encoding='utf-8') as f:
            f.write("question,answer\n")
            f.write("What is 1+1?,1+1 equals 2.\n")
            f.write("Who invented the lightbulb?,Thomas Edison is often credited with the invention of the practical incandescent light bulb.\n")
    except FileExistsError:
        pass # File already exists, proceed

    input_csv = "train.csv"
    output_jsonl = "output.jsonl"
    print(f"Converting '{input_csv}' to '{output_jsonl}'...")
    convert_csv_to_jsonl(input_csv, output_jsonl)
    print("Conversion complete.")
    print(f"Check '{output_jsonl}' for the converted data.")
