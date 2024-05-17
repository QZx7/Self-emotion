import os
import json
from agent.agent import Agent
from typing import List, Dict, Text
from utils import generate_topic


class Conversation:
    def __init__(self, initator: Agent, follower: Agent, time: Text) -> None:
        self.initiator = initator
        self.follower = follower
        self.time = time
        self.topic = ""
        self.utterances: List[Dict[Text, Text]] = []
        self.updated_by = None
        self.ended_by = None
        self.save_path = ""
        self.__initialize()

    def __initialize(self):
        topic_string, topic_dict = generate_topic(
            self.initiator.profile, self.follower.profile
        )
        self.topic = topic_string
        self.initiator.set_goal(topic_dict["goal_1"])
        self.follower.set_goal(topic_dict["goal_2"])
        self.initiator.set_topic(topic_dict["topic"])
        self.follower.set_topic(topic_dict["topic"])
        # context = f"{self.initiator.profile['name']} and {self.follower.profile['name']} are having a chitchat dialogue. They are talking about {self.topic}"
        context = f"{self.topic}"
        self.utterances.append({"context": context})

    def _save(self):
        self.path = os.path.join(os.path.dirname(__file__), "storage/conversation.json")
        save_file = open(self.path, "w+", encoding="utf-8")
        save_obj = {
            "initiator": self.initiator.name,
            "follower": self.follower.name,
            "initiator_event": self.initiator.get_current_event_string(),
            "follower_event": self.follower.get_current_event_string(),
            "time": self.time,
            "utterances": self.utterances,
        }
        json.dump(save_obj, save_file, indent=4, ensure_ascii=False)
        print(f"Conversation at {self.time}:\n{self.utterances}")

    def update_conversation(self, speaker: Agent):
        speaker.listen(self.utterances[-1])
        message = speaker.talk()
        if message:
            self.utterances.append(message)
            self.updated_by = speaker
        else:
            self.ended_by = speaker

    def is_updated_by(self, speaker: Agent):
        return True if self.updated_by == speaker else False

    def run(self):
        # both agent listen to the context firsts
        self.initiator.listen(self.utterances[-1])
        self.follower.listen(self.utterances[-1])

        # start by the initiator
        message = self.initiator.talk()
        self.utterances.append(message)
        self.updated_by = self.initiator

        # get into the loop
        while True:
            if self.is_updated_by(self.initiator):
                self.update_conversation(self.follower)
            elif self.is_updated_by(self.follower):
                self.update_conversation(self.initiator)

            if self.ended_by:
                self._save()
                break
