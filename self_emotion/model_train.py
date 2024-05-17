import os
import re
import json
import wandb
import logging
import numpy as np
import nltk
import torch
from datasets import Dataset
from dotenv import load_dotenv

os.environ["HF_HOME"] = "/work/gk77/k77033/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/work/gk77/k77033/huggingface_cache"
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

load_dotenv(".env")
evaluate_metric_name = "rouge"
metric = evaluate.load(evaluate_metric_name)


def load_data(file_path, with_se=False):
    file = open(file_path, "r", encoding="utf-8")
    lines = file.readlines()
    instances = []
    for line in lines:
        line_data = json.loads(line)
        pattern = r"\n{2,}"
        context = f"I'm having a conversation with my friend. My friend is {line_data['friend_mood']}."
        if with_se:
            context = f"I'm having a conversation with my friend. My friend is {line_data['friend_mood']} And I'm {line_data['your_mood']}"
        processed_history = re.sub(pattern, "\n", line_data["history"])
        utterances = processed_history.split("\n")
        for idx in range(0, len(utterances), 2):
            instance = {"context": "", "label": ""}
            if utterances[idx].startswith("friend:") and idx < len(utterances) - 1:
                instance["context"] = context + utterances[idx] + " </s>"
                instance["label"] = utterances[idx + 1] + "</s>"
            instances.append(instance)
    dataset = Dataset.from_list(instances)
    return dataset


def load_model(tok_name, model_name):
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    return tokenizer, model


def load_model_with_quantization(tok_name, model_name):
    os.environ["HF_HOME"] = "/work/gk77/k77033/huggingface_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/work/gk77/k77033/huggingface_cache"

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, quantization_config=config
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return result


class SavePeftModelCallback(TrainerCallback):
    """
    A Callback class that implements the saving function of Peft style models.
    """

    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def train(tokenizer, model, train_set, test_set, data_collator, save_name):
    os.environ["HF_HOME"] = "/work/gk77/k77033/huggingface_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/work/gk77/k77033/huggingface_cache"
    os.environ["WANDB_PROJECT"] = "self_emotion_2"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_API_KEY"] = "aff31a14d1ddad71f9810f0af990d98127f1b766"

    traning_args = Seq2SeqTrainingArguments(
        output_dir=f"/work/gk77/k77033/Penpal/self_emotion/models/{save_name}",
        report_to="wandb",
        logging_steps=5000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        eval_steps=5000,
        weight_decay=0.01,
        learning_rate=1e-4,
        fp16=False,
        gradient_checkpointing=False,
        num_train_epochs=3,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=traning_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        callbacks=[SavePeftModelCallback],
    )

    trainer.train()
    wandb.finish()


def tokenize_data(examples, tokenizer, max_context_length, max_target_length):
    input_encoding = tokenizer(
        examples["context"],
        padding="longest",
        truncation=True,
        max_length=max_context_length,
    )
    target_encoding = tokenizer(
        text_target=examples["label"],
        padding="longest",
        truncation=True,
        max_length=max_target_length,
    )
    labels = target_encoding.input_ids
    labels[labels == 1] = -100
    input_encoding["labels"] = labels

    return input_encoding


def inference(tokenizer, model, input_text):
    input_encoding = tokenizer(
        input_text, padding="longest", truncation=True, max_length=512
    )
    with torch.no_grad():
        tokens = model.generate(
            **input_encoding,
            max_new_token=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    output = tokenizer.decode(
        tokens[0][input_encoding["input_ids"].size(1) :], skip_special_tokens=True
    )
    print(f"raw output: {output}")
    final_response = output
    return final_response


if __name__ == "__main__":
    ### Training section
    # set task as [no_se, with_se]
    task_name = "with_se"

    logging.info("Loading datasets...")
    train_set = load_data(
        f"./self_emotion/evaluation/results/model_compare/train_{task_name}.jsonl"
    )
    test_set = load_data(
        f"./self_emotion/evaluation/results/model_compare/valid_{task_name}.jsonl"
    )

    logging.info("Dataset loaded, start loading model...")
    tok_name = "google/flan-t5-large"
    model_name = "google/flan-t5-large"

    logging.info("Loading model from %s...", model_name)
    tokenizer, model = load_model(tok_name, model_name)
    print(tokenizer.pad_token_id)
    logging.info("Model loaded, start tokenizing datset...")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    tokenized_train_set = train_set.map(
        tokenize_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_context_length": 512,
            "max_target_length": 128,
        },
        remove_columns=["context", "label"],
    )
    tokenized_test_set = test_set.map(
        tokenize_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_context_length": 512,
            "max_target_length": 128,
        },
        remove_columns=["context", "label"],
    )
    logging.info(tokenized_train_set)
    logging.info(tokenized_test_set)
    print(tokenized_train_set[:5])

    logging.info("Dataset tokenized, start training...")
    save_name = f"flan_t5_{task_name}"
    train(
        tokenizer=tokenizer,
        model=model,
        train_set=tokenized_train_set,
        test_set=tokenized_test_set,
        data_collator=data_collator,
        save_name=save_name,
    )

    ### Inference section
