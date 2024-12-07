import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TextStreamer, BitsAndBytesConfig


DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"


class Model:

    def __init__(self,
                 model_file_path,
                 streaming=True,
                 quantize="nf4",
                 torch_dtype=torch.bfloat16,
                 verbose=False,
                 debug=False):
        self.model_file_path = model_file_path
        self.streaming = streaming
        self.quantize = quantize
        self.torch_dtype = torch_dtype
        self.verbose = verbose
        self.debug = debug

        self.tokenizer = None
        self.model = None
        self.streamer = None
        
        return

    def create_quantize_config(self, quantize_type=None, torch_dtype=None):
        if torch_dtype == None:
            torch_dtype = self.torch_dtype
        if quantize_type == None:
            quantize_type = self.quantize
        if quantize_type == None:
            return None
        if quantize_type == "qint8" or quantize_type == "q8":
            return BitsAndBytesConfig(load_in_8bit=True)
        elif quantize_type == "qint4" or quantize_type == "q4":
            return BitsAndBytesConfig(load_in_4bit=True)
        elif quantize_type == "nf4" or quantize_type == "fp4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantize_type,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        return None
    
    def load_model(self):
        if self.verbose:
            print("Load tokenizer:", self.model_file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_file_path)
        
        if self.verbose:
            print("Load model:", self.model_file_path)
            print("quantize:", self.quantize)
            print("torch.dtype:", self.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_file_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype=self.torch_dtype,
            quantization_config=self.create_quantize_config()
            #attn_implementation="flash_attention_2",
        )
        self.model.eval()
        if self.streaming:
            self.streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
        return

    def query(
            self,
            prompt,
            image=None,
            history=None,
            new_max_tokens=4096,
            do_sample=True,
            temperature=0.7,
    ):
        if history != None:
            messages = history
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        if self.debug:
            print("messages:", messages)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=new_max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                streamer=self.streamer,
            )
        output = self.tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1):],
            skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": output})
        return output, messages
    
