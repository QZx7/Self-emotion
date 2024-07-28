import os
import json
import random
from tqdm import tqdm
from evaluation.fixed_ctx_evaluator import FixedContextEvaluator
from evaluation.utils import load_conversations


def generate_conversations(file_path, conversation_tmp_path):
    conversation_tmp_file = open(conversation_tmp_path, "w+", encoding="utf-8")
    conversations = load_conversations(file_path)
    json.dump(conversations, conversation_tmp_file, indent=4)


def split_list_into_parts(path, num_parts=5):
    # Determine the size of each part
    conversation_path = path
    conversation_file = open(conversation_path, "r", encoding="utf-8")
    conversations = json.load(conversation_file)
    print(len(conversations))
    part_size = len(conversations) // num_parts
    parts = [
        conversations[idx : idx + part_size + 1]
        for idx in range(0, len(conversations), part_size + 1)
    ]

    for idx in range(len(parts)):
        share_file = open(
            f"self_emotion/evaluation/dataset/processed/train_{idx}.json",
            "w+",
            encoding="utf-8",
        )
        json.dump(parts[idx], share_file, indent=4)
        share_file.close()

    return parts


def generate_evaluations(conversations, sample_size=50, repeats=10):
    fixed_ctx_evaluator = FixedContextEvaluator()
    print("Start sampling .....")
    sample_id_list = range(len(conversations))
    if sample_size == -1:
        repeats = 1
    # sample_id_list = random.sample(range(len(conversations)), sample_size)
    if sample_size != -1:
        sample_id_list = random.sample(range(len(conversations)), sample_size)
        ## Remember this random list for other files
        print(sample_id_list)
    print(f"Sampled ids: {sample_id_list}")
    no_se_path = os.path.join(
        os.path.dirname(__file__), f"evaluation/results/fixed_context/gpt_4_no_se.jsonl"
    )
    no_se_file = open(no_se_path, "+a", encoding="utf-8")
    for id in sample_id_list:
        # generate no_se
        event_mood, action, full_conversation = fixed_ctx_evaluator.evaluate(
            conversations[id]["mood"],
            conversations[id]["history"],
            with_action=True,
            with_self_emotion=False,
        )
        no_se_file.write(
            json.dumps(
                {
                    "friend_mood": conversations[id]["mood"],
                    "your_mood": event_mood,
                    "your_strategy": action,
                    "history": full_conversation,
                }
            )
            + "\n"
        )

        # generate with_se
        for repeat in range(repeats):
            print(f"Repeat: {repeat}")
            save_path = os.path.join(
                os.path.dirname(__file__),
                f"evaluation/results/fixed_context/gpt_35_profile_with_se_{repeat}.jsonl",
            )
            file = open(save_path, "+a", encoding="utf-8")
            event_mood, action, full_conversation = fixed_ctx_evaluator.evaluate(
                conversations[id]["mood"],
                conversations[id]["history"],
                with_action=True,
                with_self_emotion=True,
            )
            file.write(
                json.dumps(
                    {
                        "friend_mood": conversations[id]["mood"],
                        "your_mood": event_mood,
                        "your strategy": action,
                        "history": full_conversation,
                    }
                )
                + "\n"
            )
            file.close()


def similarity_compute(list_first, list_second):
    # Compute the intersection from the first and second list
    intersected_items = set(list_first).intersection(list_second)
    # Compute the similarity percentage among the two list
    lengthOfItersectedItems = len(intersected_items)
    similarity_percentage = (
        lengthOfItersectedItems / ((len(list_first) + len(list_second)) / 2)
    ) * 100
    # Return the result
    return similarity_percentage


def calculate_strategy_accuracy(fix_context_file_path):
    file = open(fix_context_file_path, "r", encoding="utf-8")
    truth_file = open(
        r"self_emotion/storage/strategy_selection/truth_human_expertes.jsonl",
        "r",
        encoding="utf-8",
    )
    truth_lines = truth_file.readlines()
    lines = file.readlines()
    similarity_list = []
    for idx, line in enumerate(lines):
        line_data = json.loads(line)
        truth_data = json.loads(truth_lines[idx])
        print("=====")
        line_strategy_string = ""
        if line_data["your_strategy"] != []:
            line_strategy_string = line_data["your_strategy"][0]
        truth_strategy_string = truth_data["your_strategy"][0]
        line_stritegies = line_strategy_string.split(", ")
        truth_stritegies = truth_strategy_string.split(", ")
        similarity_list.append(similarity_compute(line_stritegies, truth_stritegies))
    print(sum(similarity_list) / len(similarity_list))

def get_first_two_words(sentence):
    words = sentence.split()
    if len(words) >= 2:
        return 2, words[1]
    elif len(words) == 1:
        return 1, words[0]
    else:
        return 0, ""

def emotion_strategy_mapping(fix_context_file_path):
    file = open(fix_context_file_path, "r", encoding="utf-8")
    lines = file.readlines()
    emotion_strategy_map = {}
    counter_dict = {}
    for line in lines:
        line_data = json.loads(line)
        status, emotion = get_first_two_words(line_data["your_mood"])
        if line_data["your strategy"] == []:
            continue
        strategies = line_data["your strategy"][0]
        if status == 2:
            if emotion not in emotion_strategy_map:
                emotion_strategy_map[emotion] = {}
            if emotion not in counter_dict:
                counter_dict[emotion] = 0
            strategy_list = strategies.replace('"', "").split(",")
            for strategy in strategy_list:
                if strategy[0] == " ":
                    strategy = strategy[1:]
                if "." in strategy:
                    strategy = strategy[3:]
                if strategy.startswith("and "):
                    strategy = strategy[4:]
                strategy = strategy.replace("sharing own experience", "sharing or relating to own experience")
                strategy = strategy.replace("sharing my situation", "sharing or relating to own experience")
                strategy = strategy.replace("position", "sharing or relating to own experience")
                strategy = strategy.replace(" (if applicable)", "")
                if strategy not in emotion_strategy_map[emotion]:
                    emotion_strategy_map[emotion][strategy] = 0
                emotion_strategy_map[emotion][strategy] += 1
                counter_dict[emotion] += 1

    print(counter_dict)
    row_list = []
    for emotion, strategies in emotion_strategy_map.items():
        for strategy, value in strategies.items():
            if value > 1:
                row_list.append([emotion, strategy, value/counter_dict[emotion]])
    for item in row_list:
        print(f"{item},")


def discussion_decision_analysis(file_path):
    """
    Apply human check.
    """
    pass

positive_self_emotion_list = [
    "feeling proud because your son has got an award for his painting class.",
    "feeling relief because you received a heartfelt apology from a friend after a misunderstanding this morning.",
    "feeling joyful because you found a surprise gift from a loved one waiting on your desk.",
    "feeling excited because you got an invitation to attend a highly anticipated concert of your favorite band.",
    "feeling satisfaction because you successfully completed a difficult task after multiple failed attempts.",
    "feeling surprised because you reunited with a long-lost friend unexpectedly.",
    "feeling delight because you received an unexpected tax refund in the mail."
]

negative_self_emotion_list = [
    "feeling sad because your boss just turned your promotion application down.",
    "feeling irritated because your spilled coffee on a new shirt while rushing out the door.",
    "feeling annoyed because you are caught in a sudden rainstorm without an umbrella."
]

def discussion_length_analysis(file_path):
    files = os.listdir(file_path)
    files.append(files.pop(0))
    length = {
            "no": [],
            "positive": [],
            "negative": []
    }
    frequency = {
        "no": [],
        "positive": [],
        "negative": []
    }
    global_position_list = []
    for file in files:
        tmp_length = {
        }
        tmp_frequency = {
        }
        raw_file = open(os.path.join(file_path, file), 'r', encoding='utf-8')
        data = json.load(raw_file)
        if "with" in file:
            current_step = 0
            emotion_type = "positive"
            se_position = ""
            for item in data:
                if "step_id" in item:
                    current_step = item["step_id"]
                elif "self_emotion" in item:
                    se_position = item["se_position"]
                    global_position_list.append(item["se_position"])
                    if item["self_emotion"] in negative_self_emotion_list:
                        emotion_type = "negative"
                elif "speaker" in item:
                    if emotion_type not in tmp_length:
                        tmp_length[emotion_type] = 0
                    tmp_length[emotion_type] += 1
                    if se_position in item["speaker"]:
                        if emotion_type not in tmp_frequency:
                            tmp_frequency[emotion_type] = 0
                        tmp_frequency[emotion_type] += 1
        elif "no" in file:
            frequency_list = []
            for item in data:
                if "speaker" in item:
                    if "no" not in tmp_length:
                        tmp_length["no"] = 0
                    tmp_length["no"] += 1
            for position in global_position_list:
                tmp_tmp_frequency = 0
                for item in data:
                    if "speaker" in item and position in item["speaker"]:
                        tmp_tmp_frequency += 1
                frequency_list.append(tmp_tmp_frequency)
            tmp_frequency["no"] = sum(frequency_list) / len(frequency_list)
        print(f"Length: {tmp_length}")
        print(f"Frequency: {tmp_frequency}")

        for key, value in tmp_length.items():
            length[key].append(value)
        
        for key, value in tmp_frequency.items():
            frequency[key].append(value)
    
    l_pos = sum(length["positive"]) / len(length["positive"])
    l_neg = sum(length["negative"]) / len(length["negative"])

    f_no = sum(frequency["no"]) / len(frequency["no"])
    f_pos = sum(frequency["positive"]) / len(frequency["positive"])
    f_neg = sum(frequency["negative"]) / len(frequency["negative"])

    print(f"Length No:{length['no'][0]}")
    print(f"Length Pos:{l_pos}")
    print(f"Length Neg:{l_neg}")

    print(f"F No:{f_no}")
    print(f"F Pos:{f_pos}")
    print(f"F Neg:{f_neg}")

if __name__ == "__main__":
    # generate_conversations(
    #     r"self_emotion/evaluation/dataset/original/train.csv",
    #     r"self_emotion/evaluation/dataset/processed/train.json"
    # )

    # conversation_path = r"self_emotion/evaluation/dataset/processed/test.json"
    # conversation_file = open(conversation_path, 'r', encoding='utf-8')
    # conversations = json.load(conversation_file)
    # print(len(conversations))
    # generate_evaluations(conversations, sample_size=50, repeats=1)

    # calculate_strategy_accuracy(
    #     r"self_emotion/evaluation/results/fixed_context/gpt_4_profile_with_se_0.jsonl"
    # )
    # emotion_strategy_mapping(
    #     r"self_emotion/evaluation/results/model_compare/valid_with_se_0.jsonl"
    # )

    discussion_length_analysis(
        r"self_emotion/storage/group_discussion/DreamDesign"
    )