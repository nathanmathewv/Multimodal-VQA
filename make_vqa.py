import os
import time
import json
import dotenv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google import genai

# Load environment variables from .env file
dotenv.load_dotenv()
api_key = os.getenv("KEY_2")
client = genai.Client(api_key=api_key)

def chat_response(image_path, user_prompt, system_prompt):
    """
    Reads the image and generates a VQA response using the API.
    """
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    contents = [
        {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
                {"text": system_prompt + user_prompt}
            ]
        }
    ]
    model = "gemini-1.5-flash"
    response = client.models.generate_content(model=model, contents=contents)
    return response.text

def get_keywords(data):
    """
    Extract keywords from the listing data.
    """
    vqa_data = {}
    vqa_data["item_keywords"] = [kw.get("value", "") for kw in data.get("item_keywords", []) if kw.get("language_tag", "").startswith('en')]
    temp = vqa_data["item_keywords"].copy()
    keywords = ['color', 'product-type']
    for i in range(min(5, len(temp))):
        keywords.append(temp[i])
    return keywords

def preprocess_response(response):
    """
    Pre-process the API response to extract a JSON list of Q&A pairs.
    Expected format:
    [
      ["Question 1", "Answer 1"],
      ["Question 2", "Answer 2"]
    ]
    """
    response = response.strip()
    start_index = response.find('[')
    end_index = response.rfind(']')
    if start_index != -1 and end_index != -1:
        json_str = response[start_index:end_index+1]
        try:
            qa_pairs = json.loads(json_str)
            return qa_pairs
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
    else:
        print("No JSON array found in the response.")
    return None

def main():
    # Define range constants; adjust paths if necessary.
    SARVESH_START = 0
    SARVESH_END = 6800

    # Load the base CSV data
    if not os.path.exists("Dataset/metadata/image_data_with_vqa.csv"):
        images_data = pd.read_csv("Dataset/metadata/image_data.csv")
        images_data['vqa_response'] = None
        for index, row in images_data.iterrows():
            images_data.at[index, 'vqa_response'] = None
        images_data.to_csv("Dataset/metadata/image_data_with_vqa.csv", index=False)
    else:
        print("The CSV file already exists.")

    images_data = pd.read_csv("Dataset/metadata/image_data_with_vqa.csv")

    system_prompt = """
You are a Visual Question Answering (VQA) dataset generator.
Given an image and a list of metadata strings, generate diverse, high-quality question-answer pairs that cover visual recognition, attributes, relationships, metadata, and reasoning.
Make sure that the generated questions do not ask for any numerical answers.
Design the dataset so that people who are viewing the image are able to answer the questions.
Output each pair as a JSON response.
"""

    count = 0
    for i in range(SARVESH_START, SARVESH_END):
        if not pd.isna(images_data.iloc[i]['vqa_response']):
            print("Already generated for this image")
            continue

        time.sleep(25)
        count += 1

        image = images_data.iloc[i]['image_path']
        image_id = images_data.iloc[i]['image_id']
        print("Processing image:", image_id)

        listing = images_data.iloc[i]['listing']
        listing = json.loads(listing)
        keywords = get_keywords(listing)
        user_prompt = f"""
Image ID: {image_id}
Metadata: {keywords}
Generate 2-3 question-answer pairs.
"""

        response = chat_response(image, user_prompt, system_prompt)
        response = preprocess_response(response)
        print("Response:", response)
        print("Total images processed:", count)
        
        # Convert the response to a JSON string before saving
        images_data.at[i, 'vqa_response'] = json.dumps(response)
        images_data.to_csv("Dataset/metadata/image_data_with_vqa.csv", index=False)

if __name__ == "__main__":
    main()