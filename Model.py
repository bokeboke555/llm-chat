import os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TextStreamer, BitsAndBytesConfig


class Model:

    def __init__(self,
                 model_file_path,
                 streaming=True,
                 device_map="auto",
                 to_device=None,
                 quantize="nf4",
                 torch_dtype=torch.bfloat16,
                 verbose=False,
                 debug=False):
        self.model_file_path = model_file_path
        self.streaming = streaming
        self.device_map = device_map
        self.to_device = to_device
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
        basename, ext = os.path.splitext(self.model_file_path)
        
        if self.verbose:
            print("Load tokenizer:", self.model_file_path)
        args = {}
        if ext == ".gguf":
            args["pretrained_model_name_or_path"] = os.path.dirname(self.model_file_path)
            args["gguf_file"] = os.path.basename(self.model_file_path)
        else:
            args["pretrained_model_name_or_path"] = self.model_file_path
        if self.debug:
            print("AutoTokenizer args:", args)
        self.tokenizer = AutoTokenizer.from_pretrained(**args)
        
        if self.verbose:
            print("Load model:", self.model_file_path)
            print("device_map:", self.device_map)
            print("to_device:", self.to_device)
            print("quantize:", self.quantize)
            print("torch.dtype:", self.torch_dtype)
        args = {}
        if ext == ".gguf":
            args["pretrained_model_name_or_path"] = os.path.dirname(self.model_file_path)
            args["gguf_file"] = os.path.basename(self.model_file_path)
        else:
            args["pretrained_model_name_or_path"] = self.model_file_path
        #args["local_files_only"] = True
        if self.device_map != None:
            args["device_map"] = self.device_map
        args["torch_dtype"] = self.torch_dtype
        if self.quantize != None:
            args["quantization_config"] = self.create_quantize_config()
        if self.debug:
            print("AutoModelForCausalLM args:", args)
        self.model = AutoModelForCausalLM.from_pretrained(**args)
        if self.to_device != None:
            self.model.to(self.to_device)
        self.model.eval()
        if self.streaming:
            self.streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
        return

    def query(self,
              prompt,
              image=None,
              history=None,
              new_max_tokens=4096,
              do_sample=True,
              temperature=0.7,
              system_prompt=None,
              ):

        start_time = time.perf_counter_ns()
        
        if history != None:
            messages = history
        else:
            messages = []
            if system_prompt != None:
                messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
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
        
        end_time = time.perf_counter_ns()
        if self.verbose:
            n_tokens = len(output_ids.tolist()[0][token_ids.size(1):])
            elapsed_time = float(end_time - start_time) / 1000 /1000 / 1000
            print("Output tokens:", n_tokens)
            print("Elapsed time: {} seconds".format(elapsed_time))
            print("{} t/s".format(n_tokens / elapsed_time))
        
        return output, messages
    
