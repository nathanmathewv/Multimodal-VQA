import os
import json

def read_json_file(file_path):
    folder_path = r"Dataset/metadata"
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(content):
                try:
                    obj, new_idx = decoder.raw_decode(content, idx)
                    data.append(obj)
                    idx = new_idx
                except json.JSONDecodeError:
                    break  # stop when no further valid JSON is found
                while idx < len(content) and content[idx].isspace():
                    idx += 1

    output_file = r"Dataset/metadata/merged_listings.json"
    with open(output_file, "a", encoding="utf-8") as out:
        json.dump(data, out, indent=4)

with open(r"Dataset/metadata/merged_listings.json", "r", encoding="utf-8") as f:
    listings_dict = json.load(f)

print(len(listings_dict))