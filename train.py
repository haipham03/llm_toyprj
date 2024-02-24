import argparse

from QloraTrainer import QloraTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openlm-research/open_llama_3b_v2",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="cognitivecomputations/wizard_vicuna_70k_unfiltered",
    )
    parser.add_argument("--lora", type=str, default = None, help="Path to the pre-trained lora adapter")
    args = parser.parse_args()

    trainer = QloraTrainer(args.model, args.lora, args.data_path)

    trainer.train()