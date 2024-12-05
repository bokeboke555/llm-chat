import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TextStreamer, MllamaForConditionalGeneration


class LLM:

    def __init__(self,
                 model_file_path,
                 mllama=False,
                 verbose=False,
                 debug=False):
        self.model_file_path = model_file_path
        self.mllama = mllama
        self.verbose = verbose
        self.debug = debug

        self.tokenizer = None
        self.processor = None
        self.model = None
        self.streamer = None
        
        return

    def load_model(self):
        if self.verbose:
            print("Load tokenizer:", self.model_file_path)
        if self.mllama:
            self.processor = AutoProcessor.from_pretrained(
                self.model_file_path,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_file_path,
                #local_files_only=True,
                #device_map="auto",
                #torch_dtype="auto",
            )
        if self.verbose:
            print("Load model:", self.model_file_path)
        if self.mllama:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_file_path,
                local_files_only=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                #load_in_8bit=True,
                load_in_4bit=True,
                #attn_implementation="flash_attention_2",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_file_path,
                local_files_only=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                #torch_dtype=torch.int8,
                #load_in_8bit=True,
                load_in_4bit=True,
                #attn_implementation="flash_attention_2",
            )
        self.streamer = TextStreamer(self.tokenizer)
        return

    def chat(self):
        while True:
            line = input('> ')
            if self.debug:
                print("Input text:", line)
            if line == "quit" or line == "exit":
                break
            with torch.no_grad():
                if self.mllama:
                    messages = [
                        {"role": "user",
                         "content": [
                             {"type": "text", "text": line},
                         ]
                        },
                    ]
                    input_text = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True
                    )
                    input_ids = self.processor(
                        None,
                        input_text,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                else:
                    if True:
                        messages = [
                            {"role": "user",
                             "content": line},
                        ]
                        input_ids = self.tokenizer.apply_chat_template(
                            messages,
                            return_tensors="pt",
                            return_dict=True
                        )
                    else:
                        input_ids = self.tokenizer(
                            line,
                            return_tensors="pt"
                        )
                input_ids = input_ids.to(self.model.device)
                output_tokens = self.model.generate(
                    **input_ids,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    streamer=self.streamer,
                )
            if self.mllama:
                output = self.processor.decode(output_tokens[0], skip_special_tokens=True)
            else:
                output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            print(output)
        return

    
def main():
    parser = argparse.ArgumentParser(description="Chat for LLM.")
    parser.add_argument("--mllama", action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-X", "--debug", action='store_true')
    parser.add_argument("model")
    args = parser.parse_args()
    if args.debug:
        print("Model:", args.model)

    llm = LLM(args.model, mllama=args.mllama, verbose=args.verbose, debug=args.debug)
    llm.load_model()
    llm.chat()
    
    return


if __name__ == "__main__":
    main()
