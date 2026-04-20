from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="DeepSeek-R1 Inference API")

# MODEL_NAME = "deepseek-ai/DeepSeek-R1"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

print(f"Loading {MODEL_NAME}... This requires significant VRAM.")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the model across multiple GPUs automatically
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16, # Use bfloat16 to save memory
    trust_remote_code=True
)

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.6

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    try:
        # Tokenize input and move to the primary device
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the output
        # [len(inputs.input_ids[0]):] slices off the prompt so we only return the new text
        input_length = inputs.input_ids.shape[1]
        response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return {"response": response_text.strip()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))