import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TextStreamer, MllamaForConditionalGeneration, BitsAndBytesConfig
from Model import Model


class Mllama(Model):

    def __init__(self,
                 model_file_path,
                 quantize="nf4",
                 torch_dtype=torch.bfloat16,
                 verbose=False,
                 debug=False):
        super().__init__(
            model_file_path,
            quantize=quantize,
            torch_dtype=torch_dtype,
            verbose=verbose,
            debug=debug,
        )
        self.processor = None
        return

    def load_model(self):
        if self.verbose:
            print("Load tokenizer:", self.model_file_path)
        self.processor = AutoProcessor.from_pretrained(
            self.model_file_path,
        )
        if self.verbose:
            print("Load model:", self.model_file_path)
            print("quantize:", self.quantize)
            print("torch.dtype:", self.torch_dtype)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_file_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype=self.torch_dtype,
            quantization_config=self.create_quantize_config()
            #attn_implementation="flash_attention_2",
        )
        #self.streamer = TextStreamer(self.tokenizer)
        return

    def query(self, prompt, image=None):
        with torch.no_grad():
            if image != None:
                messages = [
                    {"role": "user",
                     "content": [
                         {"type": "image"},
                         {"type": "text", "text": prompt},
                     ]},
                ]
            else:
                messages = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": prompt},
                     ]},
                ]
            input_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            input_ids = self.processor(
                image,
                input_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = input_ids.to(self.model.device)
            output_tokens = self.model.generate(
                **input_ids,
                max_new_tokens=4096,
                #do_sample=True,
                #temperature=0.7,
                #pad_token_id=self.tokenizer.pad_token_id,
                #eos_token_id=self.tokenizer.eos_token_id,
                #streamer=self.streamer,
            )
            output = self.processor.decode(output_tokens[0], skip_special_tokens=True)
        return output
