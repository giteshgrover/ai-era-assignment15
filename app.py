import gradio as gr
import torch
from model import DeepSeekTransfomerModel
from transformers import AutoTokenizer
from config import Config
from utils import get_device
# Initialize model and tokenizer

config = Config()
device = get_device(config.seed)
print("device: ", device)

def load_model():
    model = DeepSeekTransfomerModel(config)
    # Load model weights to CPU first
    model.load_state_dict(torch.load(config.checkpoints_path + "/model_final.pt", map_location=torch.device("cpu")))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
    return model, tokenizer

model, tokenizer = load_model()  # Get device from load_model

def generate_text(input_text, max_new_tokens=100, temperature=0.8, top_k=50):
    """
    Generate text based on the input prompt
    """
    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Move output back to CPU before decoding
    output_ids = output_ids.cpu()
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label="Max New Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Text Generation",
    description="Enter a prompt and the model will generate text based on it.",
)

if __name__ == "__main__":
    demo.launch() 