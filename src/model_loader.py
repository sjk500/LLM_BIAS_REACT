import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 사용 디바이스:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)

    return model, tokenizer, device
