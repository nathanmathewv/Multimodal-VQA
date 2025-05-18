import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel
from huggingface_hub import snapshot_download

def ensure_model_available(target_dir, hf_repo_id):
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"Model found in '{target_dir}', skipping download.")
        return target_dir

    print("Fetching model weights from Hugging Face...")
    downloaded_path = snapshot_download(repo_id=hf_repo_id, local_dir=target_dir, local_dir_use_symlinks=False)

    print(f"Model files available at: {target_dir}")
    return downloaded_path


def initialize_model_and_processor(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the processor from HuggingFace, not from LoRA weights
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    # Load base model
    blip_base = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    # Apply LoRA weights
    lora_applied = PeftModel.from_pretrained(blip_base, weights_path, is_trainable=False)
    lora_applied.to(device)
    lora_applied.eval()
    
    return processor, lora_applied, device

def generate_response(image_file, query_text, processor, model, device):
    """Generate a single answer for an image-question pair."""
    try:
        img = Image.open(image_file).convert("RGB")
        prompt = f"Based on the image, answer the following question with a single word. Question: {query_text} Answer:"
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs)
            response = processor.decode(output_ids[0], skip_special_tokens=True)
        return response.split()[0].lower()
    except Exception as e:
        print(f"[Error] Skipped {image_file}: {e}")
        return "error"

def run_inference(image_folder, metadata_csv, model_dir, weights_url):
    # Prepare model
    ensure_model_available(model_dir, weights_url)
    print("Setting up model...")
    processor, model, device = initialize_model_and_processor(model_dir)

    # Read metadata
    metadata = pd.read_csv(metadata_csv)
    answers = []

    for _, entry in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
        image_path = os.path.join(image_folder, entry["image_name"])
        question = str(entry["question"])
        answer = generate_response(image_path, question, processor, model, device)
        answers.append(answer)

    metadata["generated_answer"] = answers
    metadata.to_csv("results.csv", index=False)
    print("Done! Predictions saved to results.csv")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference using LoRA fine-tuned BLIP.")
    parser.add_argument('--image_dir', required=True, help='Directory with image files')
    parser.add_argument('--csv_path', required=True, help='CSV with image_name and question columns')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    huggingface_link = "DeathlyMade/Blip-vqa-base-LoRA"
    model_dir = "./finetuned_blip_with_lora"
    run_inference(args.image_dir, args.csv_path, model_dir, huggingface_link)