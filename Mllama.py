import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TextStreamer, MllamaForConditionalGeneration, BitsAndBytesConfig
from Model import Model


class Mllama(Model):

    def __init__(self,
                 model_file_path,
                 streaming=True,
                 device_map="auto",
                 to_device=None,
                 quantize="nf4",
                 torch_dtype=torch.bfloat16,
                 verbose=False,
                 debug=False):
        super().__init__(
            model_file_path,
            streaming=streaming,
            device_map=device_map,
            to_device=to_device,
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
        if self.streaming:
            self.streamer = TextStreamer(
                self.processor,
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

        if history != None:
            messages = history
        else:
            messages = []
            if system_prompt != None:
                messages.append({"role": "system", "content": system_prompt})
        user_content = []
        if image != None:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})
        if self.debug:
            print("messages:", messages)
        if False:
            input_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
        else:
            input_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        input_ids = self.processor(
            image,
            input_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        #if self.debug:
        #    print("input_ids:", input_ids)
        with torch.no_grad():
            input_ids = input_ids.to(self.model.device)
            output_tokens = self.model.generate(
                **input_ids,
                max_new_tokens=new_max_tokens,
                #do_sample=True,
                #temperature=0.7,
                #pad_token_id=self.tokenizer.pad_token_id,
                #eos_token_id=self.tokenizer.eos_token_id,
                streamer=self.streamer,
            )
        #if self.debug:
        #    print("output_tokens:", output_tokens)
        if False:
            output = self.processor.decode(output_tokens[0], skip_special_tokens=True)
        else:
            output = self.processor.decode(
                output_tokens.tolist()[0][input_ids["input_ids"].size(1):],
                skip_special_tokens=True
            )
        messages.append({"role": "assistant", "content": output})
            
        return output, messages
