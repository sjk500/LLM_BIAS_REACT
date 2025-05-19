from src.model_loader import load_model
from src.inference import run_inference

if __name__ == "__main__":
    model_path = "/home/ksj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
    input_path = "data/test.csv"
    output_path = "output/first_submission.csv"

    model, tokenizer, device = load_model(model_path)
    run_inference(model, tokenizer, device, input_path, output_path)
