import os
import time
import json
import dotenv
import pandas as pd
from google import genai
from google.api_core import exceptions as google_exceptions

# --- Configuration ---
ENV_KEY_PREFIX = "API_KEY_" 
RETRY_DELAY_SECONDS = 5 
MAX_RETRY_WAIT_MINUTES = 2 
CSV_PATH = "Dataset/metadata/image_data_with_vqa.csv"
BASE_CSV_PATH = "Dataset/metadata/image_data.csv"
IMAGE_BASE_DIR = "" 

# --- Global State ---
api_keys = []
current_key_index = 0

def load_api_keys():
    global api_keys
    global current_key_index
    dotenv.load_dotenv()
    api_keys = []
    for key, value in os.environ.items():
        if key.startswith(ENV_KEY_PREFIX) and value:
            api_keys.append(value)
    
    if not api_keys:
        raise ValueError(f"No API keys found in .env file with prefix '{ENV_KEY_PREFIX}'")
    
    print(f"Loaded {len(api_keys)} API key(s).")
    current_key_index = 0

def get_next_key_index():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    return current_key_index

def create_genai_client(api_key):
    return genai.Client(api_key=api_key)

def chat_response(image_path, user_prompt, system_prompt):
    global current_key_index
    
    full_image_path = os.path.join(IMAGE_BASE_DIR, image_path) if IMAGE_BASE_DIR else image_path
    
    if not os.path.exists(full_image_path):
        print(f"Error: Image file not found at {full_image_path}")
        return None

    try:
        with open(full_image_path, "rb") as img_file:
            image_data = img_file.read()
    except IOError as e:
        print(f"Error reading image file {full_image_path}: {e}")
        return None

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

    initial_key_index = current_key_index
    keys_tried = 0

    while keys_tried < len(api_keys):
        current_key = api_keys[current_key_index]
        print(f"Attempting API call with key index {current_key_index}...")
        
        try:
            client = create_genai_client(current_key) 
            response = client.models.generate_content(model=model, contents=contents)
            print("API call successful.")
            return response.text

        except google_exceptions.ResourceExhausted as e:
            print(f"Rate limit error (ResourceExhausted) with key index {current_key_index}: {e}")
            keys_tried += 1
            get_next_key_index()
            print(f"Rotated to key index {current_key_index}. Waiting {RETRY_DELAY_SECONDS}s before retry...")
            time.sleep(RETRY_DELAY_SECONDS)
            
        except google_exceptions.GoogleAPIError as e:
            print(f"API Error with key index {current_key_index}: {e}")
            keys_tried += 1
            get_next_key_index()
            print(f"Rotated to key index {current_key_index} due to API error. Waiting {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)

        except Exception as e:
            print(f"Unexpected error during API call with key index {current_key_index}: {e}")
            keys_tried += 1
            get_next_key_index()
            print(f"Rotated to key index {current_key_index} due to unexpected error. Waiting {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)

    print(f"All {len(api_keys)} API keys failed due to rate limits or errors in this cycle.")
    print(f"Waiting for {MAX_RETRY_WAIT_MINUTES} minutes before trying the first key again for this image...")
    time.sleep(MAX_RETRY_WAIT_MINUTES * 60)
    
    current_key_index = initial_key_index 
    print(f"Retrying image {image_path} starting again with key index {current_key_index}")
    return None

def get_keywords(data):
    vqa_data = {}
    item_keywords_raw = data.get("item_keywords", [])
    vqa_data["item_keywords"] = [
        kw.get("value", "") for kw in item_keywords_raw 
        if isinstance(kw, dict) and kw.get("language_tag", "").startswith('en') and kw.get("value")
    ]
    temp = vqa_data["item_keywords"].copy()
    keywords = ['color', 'product-type'] 
    keywords.extend(temp[:5]) 
    return keywords

def preprocess_response(response):
    if not response: 
        return None
        
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    start_index = response.find('[')
    end_index = response.rfind(']')
    
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_str = response[start_index : end_index + 1]
        try:
            qa_pairs = json.loads(json_str)
            if isinstance(qa_pairs, list) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in qa_pairs):
                 return qa_pairs
            else:
                print(f"Warning: Decoded JSON is not in the expected format (list of pairs): {qa_pairs}")
                return None 
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Problematic JSON string part: '{json_str}'")
            print(f"Original Response content was: \n---\n{response}\n---")
    else:
        print("No JSON array found in the response.")
        print(f"Response content was: \n---\n{response}\n---")
        
    return None

def main():
    try:
        load_api_keys()
    except ValueError as e:
        print(f"Error: {e}")
        return 

    SARVESH_START = 0
    SARVESH_END = 6800 
    NATHAN_START = 6800
    NATHAN_END = 13300
    DIVYAM_START = 13300
    DIVYAM_END = 20000

    CURRENT_RANGE_START = NATHAN_START
    CURRENT_RANGE_END = NATHAN_END

    if not os.path.exists(CSV_PATH):
        try:
            images_data = pd.read_csv(BASE_CSV_PATH)
            if 'vqa_response' not in images_data.columns:
                 images_data['vqa_response'] = None
            else:
                 images_data['vqa_response'] = images_data['vqa_response'].fillna(value=pd.NA)

            print(f"Creating new CSV with vqa_response column: {CSV_PATH}")
            images_data.to_csv(CSV_PATH, index=False)
        except FileNotFoundError:
             print(f"Error: Base CSV file not found at {BASE_CSV_PATH}")
             return
        except Exception as e:
             print(f"Error initializing CSV: {e}")
             return
    else:
        print(f"Loading existing CSV file: {CSV_PATH}")
        try:
            images_data = pd.read_csv(CSV_PATH)
            if 'vqa_response' not in images_data.columns:
                print("Adding missing 'vqa_response' column to existing CSV.")
                images_data['vqa_response'] = None
        except Exception as e:
            print(f"Error loading existing CSV {CSV_PATH}: {e}")
            return

    # Keep this docstring as it defines the LLM's task
    system_prompt = """
    You are an expert Visual Question Answering (VQA) dataset creator, specializing in generating questions about product images. Your goal is to create question-answer pairs based *solely* on visual information present in an image.

    Your Constraints and Guidelines:
    *   **Visual Grounding:** All questions must be answerable by looking directly at the image provided. No external knowledge is allowed for answering.
    *   **Single-Word Answers:** Every answer must be a single, concise word (e.g., colors like "Green", confirmations like "Yes"/"No", materials like "Metal", shapes like "Square").
    *   **Diverse Questions:** Generate questions covering various visual aspects, including object identification, color, apparent material, shape, key parts, and basic spatial relationships if applicable.
    *   **Clarity:** Questions should be clear, specific, and unambiguous.
    *   **Strict Avoidance:** Do NOT generate questions about: subjective qualities, price/brand/origin unless clearly visible as large text, counting numerous small items, reading fine print, or anything requiring external data or numerical output.
    *   **Output Format:** You MUST output the results as a JSON formatted list of lists. Each inner list must contain exactly two strings: the question and the single-word answer. Example: `[["Question 1", "Answer 1"], ["Question 2", "Answer 2"]]`
        """

    processed_count = 0
    api_call_count = 0 

    actual_end = min(CURRENT_RANGE_END, len(images_data))
    print(f"Processing images from index {CURRENT_RANGE_START} to {actual_end - 1}")

    for i in range(CURRENT_RANGE_START, actual_end):
        if pd.notna(images_data.loc[i, 'vqa_response']): 
            try:
                json.loads(str(images_data.loc[i, 'vqa_response'])) # Check if it's valid JSON
                continue 
            except (json.JSONDecodeError, TypeError):
                print(f"Index {i}: Existing data is not valid JSON. Reprocessing.")

        try:
            row = images_data.iloc[i]
            image_path = row['image_path']
            image_id = row['image_id']
            listing_str = row['listing']

            if pd.isna(image_path) or pd.isna(listing_str):
                 print(f"Index {i}: Skipping due to missing image_path or listing data.")
                 continue

            listing = json.loads(listing_str) 
            keywords = get_keywords(listing)
            
        except KeyError as e:
             print(f"Index {i}: Skipping due to missing column in CSV: {e}")
             continue
        except json.JSONDecodeError as e:
             print(f"Index {i}: Skipping due to invalid JSON in 'listing' column: {e}")
             continue
        except Exception as e:
             print(f"Index {i}: Skipping due to unexpected error reading row data: {e}")
             continue

        print(f"\nProcessing Index: {i}, Image ID: {image_id} (Image Path: {image_path})")

        user_prompt = f"""
        Image ID: {image_id}
        Metadata Keywords: {keywords}
        Generate 2-3 question-answer pairs based on the image and keywords provided. Follow the JSON output format strictly.
        """
        
        print(f"Waiting {25}s before API call...") # Maintain politeness delay
        time.sleep(25)

        response_text = chat_response(image_path, user_prompt, system_prompt)
        processed_count += 1
        
        if response_text:
            api_call_count += 1
            processed_response = preprocess_response(response_text)
            print(f"Index {i}: Raw Response Text Snippet:\n{response_text[:200]}...") 
            print(f"Index {i}: Processed Response: {processed_response}")

            if processed_response:
                images_data.loc[i, 'vqa_response'] = json.dumps(processed_response)
            else:
                 print(f"Index {i}: Failed to preprocess the response. Storing None.")
                 images_data.loc[i, 'vqa_response'] = pd.NA 
        else:
            print(f"Index {i}: Failed to get a valid response after retries/errors.")
            images_data.loc[i, 'vqa_response'] = pd.NA 

        try:
            images_data.to_csv(CSV_PATH, index=False)
        except IOError as e:
            print(f"CRITICAL ERROR: Could not save CSV file: {e}. Stopping.")
            break 
        except Exception as e:
            print(f"Warning: Error saving CSV file: {e}. Continuing...")

        print(f"--- Processed in this run: {processed_count} images ---")
        print(f"--- Successful API calls in this run: {api_call_count} ---")

    print("\nProcessing finished for the selected range.")
    print(f"Total images attempted in this run: {processed_count}")
    print(f"Total successful VQA generations in this run: {api_call_count}")

if __name__ == "__main__":
    main()