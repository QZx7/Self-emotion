import re
import os
from utils import load_template, request_llm
from typing import Text, Dict

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), r"prompt_templates/experiments/group_discussion"
)
PROFILES_ROOT = os.path.join(os.path.dirname(__file__), r"./profiles/group_discussion")


class GroupDiscussionAgent:
    def __init__(
        self, backend: Text, profile: Dict[Text, Text]
    ) -> None:
        self.backend = backend
        self.profile = profile
        self.self_emotion = ""
        self.current_topic = ""
        self.current_step = ""

    def initialize_phase(self):
        if self.current_step == "":
            print("Step detail not set...")
            return
        prompt = load_template(
            path=os.path.join(TEMPLATE_PATH, r"initialize_phase.txt"),
            keys={"topic": self.current_topic, "step": self.current_step},
        )
        print(prompt)
        raw_initialize_text = request_llm("gpt-4", prompt)
        print("~~~~~~~~~~~~~~~~~~~~~~\n")
        print(raw_initialize_text)
        pattern = re.compile(r"\[BOS\](.*?)\[EOS\]", re.DOTALL)
        initialize_text = re.findall(pattern, raw_initialize_text)[0]
        initialize_text = initialize_text.replace("\n", "").replace("You: ", "")
        return initialize_text

    def talk(self, history):
        talk_template = os.path.join(TEMPLATE_PATH, r"agent_talk_no_se.txt")
        if self.self_emotion != "":
            talk_template = os.path.join(TEMPLATE_PATH, r"agent_talk_with_se.txt")
        prompt = load_template(
            path=talk_template,
            keys={
                "name": self.get_name(),
                "position": self.get_position(),
                "overview": self.get_overview(),
                "topic": self.current_topic,
                "step": self.current_step,
                "history": history,
                "self_emotion": self.self_emotion
            }
        )
        print(prompt)
        raw_response_text = request_llm("gpt-4", prompt)
        print("~~~~~~~~~~~~~~~~~~~~~~\n")
        print(raw_response_text)
        pattern = re.compile(r"\[BOS\](.*?)\[EOS\]", re.DOTALL)
        response_text = re.findall(pattern, raw_response_text)[0]
        response_text = response_text.replace("\n", "").replace("You: ", "")
        return response_text

    def get_name(self):
        return self.profile["name"]

    def get_position(self):
        return self.profile["position"]

    def get_role(self):
        return self.profile["role"]

    def get_overview(self):
        return self.profile["overview"]

    def set_topic(self, topic: Text):
        self.current_topic = topic

    def set_step(self, step: Text):
        self.current_step = step

    def set_self_emotion(self, self_emotion: Text):
        self.self_emotion = self_emotion
