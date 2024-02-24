import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, logging
import time
from tqdm import tqdm

choices = ["A", "B", "C", "D"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.set_verbosity(logging.FATAL)

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    # prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = "### HUMAN:\n" + train_prompt + prompt_end + "\n\n### RESPONSE:\nAnswer is"
        # print(prompt)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        # decoder_input_ids = model._shift_right(decoder_input_ids)
        # logits = model(
        #     input_ids=input_ids, decoder_input_ids=decoder_input_ids
        # ).logits.flatten()
        
        logits = model(
             input_ids=input_ids
        ).logits.flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        print(np.sum(cors)/len(cors))
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)

def main(args):

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = LlamaForCausalLM.from_pretrained(args.model, quantization_config=bnb_config)
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    
    base_model = LlamaForCausalLM.from_pretrained(args.model, load_in_8bit=True)
    # model = base_model
    # print("DONE LOAD MODEL")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        if subject != "miscellaneous":
            continue
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        
        print(acc, np.sum(cors)/len(cors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
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
    args = parser.parse_args()
    main(args)