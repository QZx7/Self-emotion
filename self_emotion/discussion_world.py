import re
import os
import json
import random
from utils import request_gpt, load_template
from discussion_agent import GroupDiscussionAgent
from typing import List, Text, Dict

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), r"prompt_templates/experiments/group_discussion"
)
PROFILES_PATH = os.path.join(os.path.dirname(__file__), r"profiles/group_discussion")
TOPICS_PATH = os.path.join(os.path.dirname(__file__), r"topics/group_discussion")
STORAGE_PATH = os.path.join(os.path.dirname(__file__), r"storage/group_discussion")


class GroupDiscussionWorld:
    def __init__(self, title, content) -> None:
        self.title = title
        self.content = content
        self.profiles_path = os.path.join(PROFILES_PATH, f"{self.title}")
        self.topic_path = os.path.join(TOPICS_PATH, f"{self.title}.json")
        self.storage_path = os.path.join(STORAGE_PATH, f"{self.title}")
        self.agents: Dict[Text, Dict[Text, List[GroupDiscussionAgent]] | Text | int] = {
            "members": {"leader": [], "member": []},
            "member_size": 0,
            "position_list": [],
            "name_position_list": [],
        }
        self.topic = ""
        self.steps = []
        self.history = []
        pass

    def generate_agent_profiles(self, number=6):
        """Generate the agent profiles for this group discussion.

        Args:
            number (int, optional): Number of agents. Defaults to 6.
        """
        # profiles_path = os.path.join(PROFILES_PATH, f"{self.title}")
        if os.path.exists(self.profiles_path):
            print(
                "Profiles for this group discussion exist, skipping generating profiles."
            )
            return
        os.makedirs(self.profiles_path)
        prompt = load_template(
            path=os.path.join(TEMPLATE_PATH, "generate_profile.txt"),
            keys={"content": self.content, "number": str(number)},
        )
        attempt = 0
        profiles: List[Text] = []
        while attempt < 5:
            print(f"Attempt: {attempt}")
            raw_profiles_text = request_gpt("gpt-4", prompt)
            print("~~~~~~~~~~~~~~~~~~~~~~\n")
            print(raw_profiles_text)
            pattern = re.compile(r"\[BOF\](.*?)\[EOF\]", re.DOTALL)
            profiles = re.findall(pattern, raw_profiles_text)
            if len(profiles) == number:
                break
            attempt += 1

        name_pattern = re.compile(r"Name: (.*?)\n")
        role_pattern = re.compile(r"Role: (.*?)\n")
        position_pattern = re.compile(r"Position: (.*?)\n")
        overview_pattern = re.compile(r"Overview: (.*?)\n")
        for profile in profiles:
            name = re.findall(name_pattern, profile)[0]
            role = re.findall(role_pattern, profile)[0]
            position = re.findall(position_pattern, profile)[0]
            overview = re.findall(overview_pattern, profile)[0]
            profile_file = open(
                os.path.join(self.profiles_path, f"{name}.json"), "w+", encoding="utf-8"
            )
            json.dump(
                {
                    "name": name,
                    "role": role,
                    "position": position,
                    "overview": overview,
                },
                profile_file,
                indent=4,
            )

    def generate_topic_steps(self, topic: Text):
        """
        Generate the steps to of the current topic.

        Args:
            topic (Text): The text representation of the topic.
        """
        position_list = []
        for member_profile_path in os.listdir(self.profiles_path):
            member_profile_file = open(
                os.path.join(self.profiles_path, member_profile_path),
                "r",
                encoding="utf-8",
            )
            member_profile: Dict[Text, Text] = json.load(member_profile_file)
            position_list.append(member_profile["position"])
        member_size = len(os.listdir(self.profiles_path))
        if os.path.exists(self.topic_path):
            print("Topic for this group discussion exist, skipping generating topics.")
            return
        prompt = load_template(
            path=os.path.join(TEMPLATE_PATH, r"generate_steps.txt"),
            keys={
                "content": self.content,
                "number": str(member_size),
                "topic": topic,
                "position_list": str(position_list),
            },
        )

        attempt = 0
        while attempt < 5:
            raw_steps_text = request_gpt("gpt-4", prompt)
            print("~~~~~~~~~~~~~~~~~~~~~~\n")
            print(raw_steps_text)
            pattern = re.compile(r"\[BOF\](.*?)\[EOF\]", re.DOTALL)
            steps = re.findall(pattern, raw_steps_text)
            if "" not in steps:
                break
            attempt += 1
        content_pattern = re.compile(r"Content: (.*?)\n")
        active_member_pattern = re.compile(r"Active members: (.*?)\n")
        content = re.findall(content_pattern, raw_steps_text)
        active_members = re.findall(active_member_pattern, raw_steps_text)
        topic_file = open(self.topic_path, "w+", encoding="utf-8")
        steps: Dict[Text, Text | List[Dict]] = {"topic": topic, "steps": []}
        for index in range(len(content)):
            steps["steps"].append(
                {"content": content[index], "active_members": active_members[index]}
            )
        json.dump(steps, topic_file, indent=4)

    def _get_agent_by_position(self, position):
        for _, value in self.agents["members"].items():
            for member in value:
                if member.get_position() == position:
                    return member

    def _get_agent_by_name(self, name):
        for _, value in self.agents["members"].items():
            for member in value:
                if member.get_name() == name:
                    return member

    def _get_current_history_text(self):
        history_text = ""
        for utterance in self.history:
            if "speaker" not in utterance:
                continue
            history_text += f"{utterance['speaker']}: {utterance['utterance']}\n"
        return history_text

    def initialize_group(self):
        """
        Load a group of people as agents from the profiles of the current group.
        """
        if not os.path.exists(self.profiles_path) or not os.path.exists(
            self.topic_path
        ):
            print(
                "Can't initiate the group because either the profiles or the topic is missing. Run generate_agent_profiles and generate_topic_steps before initializing the group."
            )
            return
        position_list = []
        member_size = 0
        name_position_list = []
        for member_profile_path in os.listdir(self.profiles_path):
            member_profile_file = open(
                os.path.join(self.profiles_path, member_profile_path),
                "r",
                encoding="utf-8",
            )
            member_profile: Dict[Text, Text] = json.load(member_profile_file)
            if member_profile["role"].lower() == "leader":
                self.agents["members"]["leader"].append(
                    GroupDiscussionAgent("gpt-4", member_profile)
                )
            elif member_profile["role"].lower() == "member":
                self.agents["members"]["member"].append(
                    GroupDiscussionAgent("gpt-4", member_profile)
                )
            member_size += 1
            position_list.append(member_profile["position"])
            name_position_list.append(
                f"{member_profile['name']}: {member_profile['position']}"
            )
        self.agents["member_size"] = member_size
        self.agents["position_list"] = position_list
        self.agents["name_position_list"] = name_position_list
        for key, value in self.agents["members"].items():
            for member in value:
                print(f"{key}: {member.get_name()}")

        topic_step_file = open(self.topic_path, "r", encoding="utf-8")
        topic_step = json.load(topic_step_file)
        self.topic = topic_step["topic"]
        self.steps = topic_step["steps"]

    def get_next_speaker(self, step_content):
        prompt = load_template(
            path=os.path.join(TEMPLATE_PATH, r"decide_next_speaker.txt"),
            keys={
                "topic": self.topic,
                "step": step_content,
                "position_list": str(self.agents["position_list"]),
                "history": self._get_current_history_text(),
            },
        )
        raw_next_speaker_text = request_gpt("gpt-4", prompt)
        print("~~~~~~~~~~~~~~~~~~~~~~\n")
        print(raw_next_speaker_text)
        pattern = re.compile(r"\[BOS\](.*?)\[EOS\]", re.DOTALL)
        next_speaker = re.findall(pattern, raw_next_speaker_text)[0]
        next_speaker = next_speaker.replace("\n", "").replace("Next speaker: ", "")
        return self._get_agent_by_position(next_speaker)

    def update_history(self, agent: GroupDiscussionAgent, is_initial=False):
        utterance = ""
        if is_initial:
            utterance = agent.initialize_phase()
        else:
            utterance = agent.talk(self._get_current_history_text())
        self.history.append(
            {
                "speaker": f"{agent.get_name()} ({agent.get_position()})",
                "utterance": utterance,
            }
        )

    def run_discussion(self, self_emotion="", number=0):
        """
        Run the discussion.
        """
        if not os.path.exists(self.storage_path):
            os.mkdir(self.storage_path)
        discussion_save_path = os.path.join(self.storage_path, r"discussion_no_se.json")
        se_position = ""
        if self_emotion != "":
            discussion_save_path = os.path.join(self.storage_path, f"discussion_with_se_{number}.json")
            se_position = random.sample(self.agents["position_list"], 1)[0]
            self.history.append({"se_position": se_position, "self_emotion": self_emotion})
        discussion_save_file = open(discussion_save_path, 'w+', encoding='utf-8')
        for idx, step in enumerate(self.steps):
            self.history.append({"step_id": idx + 1})
            for _, value in self.agents["members"].items():
                for member in value:
                    member.set_topic(self.topic)
                    member.set_step(step["content"])
                    if member.get_position() == se_position:
                        member.set_self_emotion(self_emotion)

            # initialize
            self.update_history(self.agents["members"]["leader"][0], is_initial=True)
            # decide the next speaker
            turn = 0
            while turn <= 8:
                next_speaker = self.get_next_speaker(step["content"])
                if not next_speaker:
                    break
                self.update_history(next_speaker)
                turn += 1
        json.dump(self.history, discussion_save_file, indent=4)
        # clean history after one discussion
        self.history = []
        # reset self_emotion
        self._get_agent_by_position(se_position).set_self_emotion("")
        discussion_save_file.close()

if __name__ == "__main__":
    # dream_design = GroupDiscussionWorld(
    #     title="DreamDesign", content="DreamDesign is an excellent house design team"
    # )
    dream_design = GroupDiscussionWorld(
        # title="DreamDesign", content="DreamDesign is an excellent house design team"
        # title="HR Group-3", content="As a team of a financial company, HR Group-3 is the HR team focusing on scouting talented employees"
        # title="FantSoftware", content="FantSoftware is a newly created software development team"
        # title="KidsCare", content="KidsCare is a team devoting to charity activities for kids with disabilities"
        title="HighSchoolAlumni", content="HighSchoolAlumni is a group of students graduated from North River Bank High school"
    )
    # dream_design.generate_agent_profiles(number=6)
    # dream_design.generate_topic_steps("hosting a welcome party for all new employees and provide new members good experience in a hybrid way given the ongoing pandemic")
    # dream_design.generate_topic_steps("building a house and gain as much profits as possible out of a very limited budget")
    dream_design.initialize_group()

    self_emotion_list = [
        "feeling sad because your boss just turned your promotion application down.",
        "feeling proud because your son has got an award for his painting class.",
        "feeling relief because you received a heartfelt apology from a friend after a misunderstanding this morning.",
        "feeling joyful because you found a surprise gift from a loved one waiting on your desk.",
        "feeling irritated because your spilled coffee on a new shirt while rushing out the door.",
        "feeling excited because you got an invitation to attend a highly anticipated concert of your favorite band.",
        "feeling annoyed because you are caught in a sudden rainstorm without an umbrella.",
        "feeling satisfaction because you successfully completed a difficult task after multiple failed attempts.",
        "feeling surprised because you reunited with a long-lost friend unexpectedly.",
        "feeling delight because you received an unexpected tax refund in the mail."
    ]

    # dream_design.run_discussion()
    for idx, item in enumerate(self_emotion_list):
        dream_design.run_discussion(
            self_emotion=item,
            number=idx
        )
