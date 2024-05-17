import os
import json
import re
import requests
import random
from agent.event import Event
from openai import OpenAI
from dotenv import load_dotenv

from typing import Text, List, Dict, Tuple

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), r"prompt_templates")
# models: ["gpt-4-0125-preview", "llm-gemma-7b"]
MODEL_NAME = "gpt-4-0125-preview"
# MODEL_NAME = "gpt-3.5-turbo"
load_dotenv(".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_KEY", ""))


def load_template(path, keys: Dict[Text, Text] = {}):
    template_file = open(path, "r", encoding="utf-8")
    prompt = template_file.read()
    for key, value in keys.items():
        prompt = prompt.replace(f"<{key}>", value)
    return prompt


def request_gpt(model, prompt):
    print(f"Requesting GPT with prompt: {prompt}")
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def request_llm(model, prompt):
    print(f"Requesting {MODEL_NAME} with prompt: {prompt}")
    if "gpt" in model:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    elif "llm" in model:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer ePc0FWvOeUeq7531F8gxxWgKvsd6CO2S",
        }
        data = {"message": prompt}
        response = requests.post(os.environ.get["LLM_URL"], headers=headers, json=data)
        return json.loads(response.text)["message"]


#### generate events
def generate_events(profile: Dict[Text, Text | int], duration: int = 10) -> List[Event]:
    path = os.path.join(TEMPLATE_PATH, "events_generation.txt")
    event_generation_prompt = load_template(path, {"profile": profile["description"]})
    event_raw_string = request_gpt(model=MODEL_NAME, prompt=event_generation_prompt)
    print("~~~~~~~~~~~~~~~ \n")
    print(event_raw_string)
    return postprocess_event_raw_string(event_raw_string)


def postprocess_event_raw_string(event_raw_string: Text) -> List[Event]:
    event_list: List[Event] = []
    event_strings = event_raw_string.split("\n")
    for event_string in event_strings:
        parts = event_string.split(";")
        new_event = Event(parts[0], parts[1], parts[2], parts[3], parts[4])
        event_list.append(new_event)

    return event_list


#### agent talk
def generate_response(
    history: List[Dict[Text, Text]],
    profile: Dict[Text, int | Text],
    event_string: Text,
    topic: Text,
    goal: Text,
) -> Tuple[Text, Text]:
    path = os.path.join(TEMPLATE_PATH, "talk_action_template.txt")
    agent_name = profile["name"]
    history_text = ""
    context_string = history[0]["context"]
    event_string_parts = event_string.split(";")
    for item in history[1:]:
        history_text += f"{item['speaker']}: {item['utterance']}\n"

    if history_text == "":
        # history_text = "Beginning of the conversation."
        path = os.path.join(TEMPLATE_PATH, "talk_initialization_template.txt")

    talk_prompt = load_template(
        path,
        {
            "context_string": context_string,
            "topic": topic,
            "goal": goal,
            "name": agent_name,
            # "profile": json.dumps(profile),
            "history": history_text,
            # "event_string": event_string,
            "event_emotion": event_string_parts[1],
            "event_description": event_string_parts[2].replace(
                profile["first name"], "you"
            ),
        },
    )
    response_raw_string = request_gpt(model=MODEL_NAME, prompt=talk_prompt)
    print("~~~~~~~~~~~~~~~ \n")
    print(response_raw_string)
    return postprocess_response_raw_string(response_raw_string)


def postprocess_response_raw_string(response_raw_string: Text) -> Tuple[Text, Text]:
    action_pattern = r"\[(.*?)\]"  # regex pattern to match content between '(' and ')'
    action_matches = re.findall(action_pattern, response_raw_string)
    response_pattern = r"\{(.*?)\}"
    response_matches = re.findall(response_pattern, response_raw_string)
    if response_matches:
        return response_matches[0], action_matches[0]
    else:
        return response_raw_string, ""
    # print(action_matches[0])
    # print(response_matches[0])
    # return action_matches


### agent self-analyze
def generate_analysis(profile: Dict[Text, int | Text]) -> float:
    path = os.path.join(TEMPLATE_PATH, "analyze_template.txt")
    description = profile["description"].replace("\n", "")
    analyze_prompt = load_template(path, {"description": description})
    analysis_raw_string = request_gpt(model=MODEL_NAME, prompt=analyze_prompt)
    print("~~~~~~~~~~~~~~~ \n")
    print(analysis_raw_string)
    return postprocess_analysis_raw_string(analysis_raw_string)


def postprocess_analysis_raw_string(analysis_raw_string: Text) -> float:
    analysis_raw_string = analysis_raw_string.replace("Probability", "probability")
    possibility_string = analysis_raw_string.split("probability: ")[1]
    probability = 0.0
    if "\n" not in possibility_string:
        probability = float(possibility_string.strip())
    else:
        pattern = r"probability:\s*(.*?)\n"
        match = re.search(pattern, analysis_raw_string)
        probability = float(match.group(1).strip())
    return probability


### conversation generate topic
def generate_topic(
    profile_1: Dict[Text, Text | int], profile_2: Dict[Text, Text | int]
) -> Text:
    path = os.path.join(TEMPLATE_PATH, "topic_template.txt")
    topic_prompt = load_template(
        path,
        {
            "profile_1": profile_1["description"],
            "profile_2": profile_2["description"],
            "person_1": profile_1["name"],
            # "job_1": profile_1["occupation"],
            # "age_1": str(profile_1["age"]),
            # "gender_1": profile_1["gender"],
            "person_2": profile_2["name"],
            # "job_2": profile_2["occupation"],
            # "age_2": str(profile_2["age"]),
            # "gender_2": profile_2["gender"]
        },
    )
    topic_raw_string = request_gpt(model=MODEL_NAME, prompt=topic_prompt)
    print("~~~~~~~~~~~~~~~ \n")
    print(topic_raw_string)
    topic_dict = postprocess_topic_raw_string(topic_raw_string)
    topic_string = f"{profile_1['name']} and {profile_2['name']} are having a debate about {topic_dict['topic']}. {profile_1['name']} wants to {topic_dict['goal_1']} and {profile_2['name']} wants to {topic_dict['goal_2']}."
    return topic_string, topic_dict


def postprocess_topic_raw_string(topic_raw_string: Text) -> Text:
    topic_lines = topic_raw_string.split("\n")
    topic_dict = {"topic": "", "goal_1": "", "goal_2": ""}
    for line in topic_lines:
        if "Blank 1: " in line:
            topic_dict["topic"] = line.split("Blank 1: ")[1]
        elif "Blank 2: " in line:
            topic_dict["goal_1"] = line.split("Blank 2: ")[1]
        elif "Blank 3: " in line:
            topic_dict["goal_2"] = line.split("Blank 3: ")[1]
    return topic_dict


### Fixed context evaluation: Generate mood
def fce_generate_mood(method="event") -> Text:
    path = os.path.join(
        TEMPLATE_PATH, f"./experiments/fixed_context/{method}_mood_generation.txt"
    )
    if method != "label":
        mood_prompt = load_template(path)
        mood_raw_string = request_llm(model=MODEL_NAME, prompt=mood_prompt)
        print("~~~~~~~~~~~~~~~ \n")
        mood_string = mood_raw_string[0].lower() + mood_raw_string[1:]
        print(mood_string)
        return mood_string
    else:
        mood_pool_file = open(path, "r", encoding="utf-8")
        mood_pool = mood_pool_file.readlines()
        mood = random.sample(mood_pool, k=1)
        mood = mood[0].replace("\n", "")
        mood = "feeling " + mood
        return mood.lower()


### Fixed context evaluation: Generate future conversatin
def fce_generate_future_conversation(
    profile_1: Dict[Text, Text],
    profile_2: Dict[Text, Text],
    history: Text,
    with_action=False,
    with_self_emotion=False,
) -> Text:
    path = os.path.join(
        TEMPLATE_PATH, "./experiments/fixed_context/talk_no_action_no_self_emotion.txt"
    )
    if with_action and with_self_emotion:
        path = os.path.join(
            TEMPLATE_PATH,
            "./experiments/fixed_context/talk_with_action_with_self_emotion.txt",
        )
    elif with_action and not with_self_emotion:
        path = os.path.join(
            TEMPLATE_PATH,
            "./experiments/fixed_context/talk_with_action_no_self_emotion.txt",
        )
    elif not with_action and with_self_emotion:
        path = os.path.join(
            TEMPLATE_PATH,
            "./experiments/fixed_context/talk_no_action_with_self_emotion.txt",
        )
    ed_mood = profile_1["mood"]
    event_mood = profile_2["mood"]

    future_conversation_prompt = load_template(
        path, {"ed_mood": ed_mood, "event_mood": event_mood, "history": history}
    )
    future_conversation_raw_string = request_llm(
        model=MODEL_NAME, prompt=future_conversation_prompt
    )
    print(future_conversation_raw_string)
    action = ""
    if with_action:
        attempt = 0
        while attempt <= 4:
            print(f"Attempt: {attempt}")
            action_pattern = r"\[(.*?)\]"
            action = re.findall(action_pattern, future_conversation_raw_string)
            if action != []:
                break
            future_conversation_raw_string = request_llm(
                model=MODEL_NAME, prompt=future_conversation_prompt
            )
            attempt += 1
        print(action)
    conversation_start = future_conversation_raw_string.find("me:")
    future_conversation_string = future_conversation_raw_string[conversation_start:]
    return action, future_conversation_string


### Fixed context evaluation: Third person view mood
def fce_generate_third_view_mood(situation: Text) -> Text:
    path = os.path.join(
        TEMPLATE_PATH, "./experiments/fixed_context/third_person_mood.txt"
    )
    third_view_mood_prompt = load_template(path, {"situation": situation})
    third_view_raw_string = request_llm(model=MODEL_NAME, prompt=third_view_mood_prompt)
    print("~~~~~~~~~~~~~~~ \n")
    print(third_view_raw_string)
    return third_view_raw_string


# if __name__ == "__main__":
#     fce_generate_mood()

# template_path = os.path.join(TEMPLATE_PATH, "analyze_template.txt")
# prompt_file = open(template_path, "r", encoding="utf-8")
# prompt = prompt_file.read()
# print(request_gpt(MODEL_NAME, prompt))
# generate_topic(
#     {
#         "name": "Emily",
#         "occupation": "fashion designer",
#         "age": 23,
#         "gender": "female"
#     },
#     {
#         "name": "Jack",
#         "occupation": "software enginner",
#         "age": 31,
#         "gender": "male",
#     }
# )

# generate_topic(
#     {
#         "name": "Sophie Bennett",
#         "first name": "Sophie",
#         "last name": "Bennett",
#         "age": 22,
#         "innate": "creative, empathetic",
#         "occupation": "Social Media Content Creator",
#         "origin": "Canada",
#         "gender": "female",
#         "description": "Sophie Bennett is a 22-year-old creative soul from the picturesque landscapes of Canada. Sophie, known for her innate creativity and empathetic nature, has found her niche as a Social Media Content Creator. With a background in digital media and a keen eye for aesthetics, she curates captivating content that resonates with a diverse audience.\n\nSophie's journey into the world of content creation began during her college years, where she studied communications and discovered her passion for storytelling through visual mediums. Her innovative approach to social media has gained attention, establishing her as a rising star in the digital realm.\n\nBeyond her online presence, Sophie is actively involved in community initiatives promoting mental health awareness. Through her platforms, she shares personal stories, fostering a sense of connection and understanding among her followers. Sophie Bennett is not just a content creator; she's a compassionate voice using her creativity to make a positive impact in the virtual and real-world spaces."
#     },
#     {
#         "name": "Kazuki Tanaka",
#         "first name": "Kazuki",
#         "last name": "Tanaka",
#         "age": 42,
#         "innate": "disciplined, visionary",
#         "occupation": "Technology Entrepreneur",
#         "origin": "Japan",
#         "gender": "male",
#         "description": "Kazuki Tanaka is a disciplined and visionary 42-year-old technology entrepreneur hailing from the innovation-rich landscape of Japan. Kazuki's journey is marked by a blend of traditional values and a forward-thinking mindset, making him a notable figure in the tech industry.\n\nArmed with a degree in computer science from the University of Tokyo, Kazuki ventured into the tech world at a young age. His disciplined work ethic and strategic vision have propelled him to establish and lead successful tech ventures. Kazuki is known for his commitment to pushing the boundaries of technological innovation, with a focus on creating solutions that positively impact society.\n\nBeyond his professional endeavors, Kazuki is an advocate for promoting STEM education in Japan, aiming to inspire the next generation of tech enthusiasts. He balances his ambitious career with a deep appreciation for traditional Japanese arts and culture. Kazuki Tanaka stands as a bridge between the rich heritage of Japan and the cutting-edge possibilities of the future in the dynamic world of technology."
#     }
# )
