import gc
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def safe_model_name(model_id):
    return model_id.replace("/", "_")


def safe_mode_name(mode):
    return mode.replace("/", "_")


def cleanup_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(model_id, force_no_quant=False):
    SMALL_MODEL_THRESHOLD = 1_000_000_000

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        dummy = AutoModelForCausalLM.from_config(config)
    n_params = sum(p.numel() for p in dummy.parameters())
    del dummy

    use_quantization = n_params >= SMALL_MODEL_THRESHOLD and not force_no_quant

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        print(f"Loading model {model_id} ({n_params/1e9:.1f}B params) in 8-bit mode...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif n_params >= SMALL_MODEL_THRESHOLD:
        print(f"Loading model {model_id} ({n_params/1e9:.1f}B params) in bf16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print(f"Loading model {model_id} ({n_params/1e6:.0f}M params) in fp32...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )

    device = next(model.parameters()).device
    model.eval()
    print(f"Model loaded on device: {device}")

    return model, tokenizer, device