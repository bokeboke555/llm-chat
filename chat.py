import argparse
import torch
from PIL import Image
from Model import Model
from Mllama import Mllama


def chat(model,
         image=None,
         new_max_tokens=4096,
         do_sample=True,
         temperature=0.7,
         verbose=False,
         debug=False):
    if verbose:
        print("Image:", image)
        print("New max tokens:", new_max_tokens)
        print("Do sample:", do_sample)
        print("Temperature:", temperature)

    history = None
    while True:
        line = input('> ')
        if debug:
            print("Input text:", line)
        if line == "quit" or line == "exit":
            break
        output, history = model.query(
            line,
            image=image,
            history=history,
            new_max_tokens=new_max_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        if not model.streaming:
            print(output)
        if debug:
            print("Output:", output)
            print("history:", history)
    return

    
def main():
    parser = argparse.ArgumentParser(description="Chat for LLM.")
    parser.add_argument("--mllama", action='store_true')
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16"
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "q8", "q4", "nf4", "fp4"],
        default="nf4"
    )
    parser.add_argument("--no-streaming", action='store_true')
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--do-not-sample", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("-i", "--image")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-X", "--debug", action='store_true')
    parser.add_argument("model")
    
    args = parser.parse_args()
    
    do_sample = True
    if args.do_not_sample:
        do_sample = False
    streaming = True
    if args.no_streaming:
        streaming = False
    image = None
    if args.image != None:
        image = Image.open(args.image)

    torch_dtype = torch.bfloat16
    if args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "float32":
        torch_dtype = torch.float32

    quantize = args.quantize
    if quantize == "none":
        quantize = None
        
    if args.debug:
        print("Model:", args.model)
        print("Mllama model:", args.mllama)
        print("Image path:", args.image)
        print("Streaming:", streaming)
        print("Max tokens:", args.max_tokens)
        print("Do sample:", do_sample)
        print("temperature:", args.temperature)

    model_args = {}
    model_args["model_file_path"] = args.model
    model_args["streaming"] = streaming
    model_args["quantize"] = quantize
    model_args["torch_dtype"] = torch_dtype
    model_args["verbose"] = args.verbose
    model_args["debug"] = args.debug
    if args.mllama:
        model = Mllama(**model_args)
    else:
        model = Model(**model_args)
    model.load_model()
    chat_args = {}
    chat_args["model"] = model
    chat_args["image"] = image
    chat_args["new_max_tokens"] = args.max_tokens
    chat_args["do_sample"] = do_sample
    chat_args["temperature"] = args.temperature
    chat_args["verbose"] = args.verbose
    chat_args["debug"] = args.debug
    chat(**chat_args)
    
    return


if __name__ == "__main__":
    main()
