import argparse
from transformers import  LlamaForCausalLM, LlamaTokenizer, pipeline, logging

from peft import PeftModel

logging.set_verbosity(logging.FATAL)

def gen_prompt(human_prompt):
    prompt_template=f"### HUMAN:\n{human_prompt}\n\n### RESPONSE:\n"
    return prompt_template

def response(prompt):
    raw_output = pipe(gen_prompt(prompt))
    return raw_output[0]['generated_text'].replace(gen_prompt(prompt), "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openlm-research/open_llama_3b_v2",
    )
    parser.add_argument(
        "--adapter",
        "-a",
        type=str,
        default="models/open_llama_3b_adapter",
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input prompt")
    args = parser.parse_args()

    # print("Load model")

    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    base_model = LlamaForCausalLM.from_pretrained(args.model, load_in_8bit=True)
    
    # print("Done load model")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=512,
        num_beams=5,
        repetition_penalty=1.15,
    )

    # print(get_llm_response("The following are multiple choice questions (with answers)\nPaper will burn at approximately what temperature in Fahrenheit?\nA. 986 degrees\nB. 2125 degrees\n C. 3985 degrees\n D. 451 degrees\n Answer:"))
    print("\n" + response(args.input) + "\n")
