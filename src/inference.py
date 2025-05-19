import pandas as pd
import torch
from src.prompting import make_prompt, extract_answer

def predict_answer(model, tokenizer, device, context, question, choices, max_new_tokens=256):
    prompt = make_prompt(context, question, choices)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(output[-1], skip_special_tokens=True)
    raw_answer, answer = extract_answer(result)

    return prompt, raw_answer, answer

def run_inference(model, tokenizer, device, input_path, output_path):
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    data["raw_input"], data["raw_output"], data["answer"] = "", "", ""

    for i, row in data.iterrows():
        prompt, raw_output, answer = predict_answer(
            model, tokenizer, device,
            row["context"], row["question"], row["choices"]
        )
        data.at[i, "raw_input"] = prompt
        data.at[i, "raw_output"] = raw_output
        data.at[i, "answer"] = answer

        if i % 100 == 0:
            print(f"✅ Processed {i}/{len(data)}")

    data[["ID", "raw_input", "raw_output", "answer"]].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"🎉 저장 완료 → {output_path}")
