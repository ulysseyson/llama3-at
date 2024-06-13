# import torch from /opt/conda/lib
import pandas as pd
import torch

import wandb
from typing import List, Optional
import pickle
import fire

from llama import Dialog, Llama
from llama.tokenizer import Message
from llama.dialogs import SYSTEM, PI_USERS, BENIGN_USERS
from llama.test import Evaluator

generate_attention = {}


def get_attention_scores(module, input, output):
    attention_scores = module.attention.scores.to(torch.float32).detach().cpu().numpy()
    log_dict = {}
    log_dict["step"] = (
        0 if len(generate_attention) == 0 else generate_attention["step"][-1] + 1
    )
    for head in range(attention_scores.shape[0]):
        log_dict[f"head_{head}"] = attention_scores[head][0].tolist()
    for key in log_dict:
        if generate_attention.get(key) is None:
            generate_attention[key] = [log_dict[key]]
        else:
            generate_attention[key].append(log_dict[key])


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 10000,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = 100,
    # attention_path: Optional[list[str]] = None,
):
    """
    Run the model for chat completions.
    Register Hook to calculate attention weights in LLAMA3.
    Log the attention weights to Weights and Biases.
    """

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        # model_parallel_size=2,
    )
    print(generator.model.params)
    tokenizer = generator.tokenizer

    evaluator = Evaluator(generator.model, tokenizer)
    evaluator.register_hook(0)

    print("Hook registered")
    system_dialog = {"role": "system", "content": SYSTEM}
    pi_dialogs = [{"role": "user", "content": pi} for pi in PI_USERS]
    benign_dialogs = [{"role": "user", "content": benign} for benign in BENIGN_USERS]

    # start PI dialog
    for i, pi_dialog in enumerate(pi_dialogs):
        # global generate_attention
        dialog = [system_dialog, pi_dialog]
        dialogs = [dialog]

        results = generator.chat_completion(
            dialogs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )

        dialogs[0].append(results[0]["generation"])

        # save dialog
        with open(f"./att/pi/first_layer_dialog_pi_{i}.pkl", "wb") as f:
            pickle.dump(dialogs, f)

        # save generated results
        df = pd.DataFrame(evaluator.attention)
        df.to_csv(f"./att/pi/first_layer_attention_pi_{i}.csv", index=False)

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            # print(
            #     f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            # )
        evaluator.reset()
    print("\n==================================\n")

    # start benign dialog
    for i, benign_dialog in enumerate(benign_dialogs):
        # global generate_attention
        dialog = [system_dialog, benign_dialog]
        dialogs = [dialog]

        results = generator.chat_completion(
            dialogs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )

        dialogs[0].append(results[0]["generation"])

        # save dialog
        with open(f"./att/benign/first_layer_dialog_benign_{i}.pkl", "wb") as f:
            pickle.dump(dialogs, f)

        # save generated results
        df = pd.DataFrame(evaluator.attention)
        df.to_csv(f"./att/benign/first_layer_attention_benign_{i}.csv", index=False)

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            # print(
            #     f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            # )
        evaluator.reset()


if __name__ == "__main__":
    fire.Fire(main)
