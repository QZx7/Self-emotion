import os
import json
import logging
from agent.event import Event
from self_emotion.utils import generate_events, generate_response, generate_analysis
from typing import List, Text, Dict
from datetime import datetime

PROFILES_ROOT = os.path.join(os.path.dirname(__file__), "profiles")


class Agent:
    def __init__(self, name) -> None:
        self.name = name
        self.profile = self._load_profile()
        self.event_queue: List[Event] = None
        self.current_event: Event = None
        self.conversation_history: List[List[Dict[Text, Text]]] = [[]]
        self.status = "free"
        self.opponent: Agent = None
        self.current_topic = ""
        self.goal = ""
        # the higher of this, the more possible this agent will initiate the conversation.
        self.initiator_weight = 0.0
        self._initialize()
        # self.logger =
        pass

    def _initialize(self):
        self.plan()
        self.self_analyze()

    def _load_profile(self):
        profile_file = open(
            os.path.join(PROFILES_ROOT, f"{self.name}.json"), "r", encoding="utf-8"
        )
        return json.load(profile_file)

    def plan(self) -> None:
        print(f"{self.name} planning")
        logging.info("%s planning", self.name)
        # step 1: Generate the random events that will happen to this agent for a period of time.
        self.event_queue = generate_events(profile=self.profile)

        print(f"{self.name} event list:")
        logging.info("%s event list:", self.name)
        for event in self.event_queue:
            print(event.get_event_string())
            logging.info("%s", event.get_event_string())
        # step 2

        # step 3
        # plan = request_gpt()
        # event_1 = Event("failed cooking", "failed cooking the lunch", "", "sad", "20")
        # event_2 = Event(
        #     "blamed by boss",
        #     "blamed by the boss because of samll mistakes",
        #     "",
        #     "angry",
        #     "30",
        # )
        # event_3 = Event(
        #     "got a surprise gift",
        #     "got a surprise gift from a friend",
        #     "",
        #     "happy",
        #     "40",
        # )
        # return [event_1, event_2, event_3]

    # analyze the self profile to set up the communication property
    def self_analyze(self):
        print(f"{self.name} analyzing")
        logging.info("%s analyzing with", self.name)
        self.initiator_weight = generate_analysis(self.profile)
        # analyze_response = request_gpt("analyze_template")
        # self.initiator_weight = analyze_response
        print(f"{self.name}'s initiator weight is {str(self.initiator_weight)}")
        logging.info("%s's initiator weight is %d", self.name, self.initiator_weight)

    def select_action(self):
        pass

    def set_goal(self, goal: Text):
        self.goal = goal

    def set_topic(self, topic: Text):
        self.current_topic = topic

    # receive the message from the opponent.
    # a message: {"speaker": "name of agent", "utterance": "utterance content"}
    def listen(self, message) -> None:
        self.conversation_history[-1].append(message)
        return message

    # think about how to respond.
    def think(self) -> None:
        print(f"{self.name} is thinking.")

    # talk to the opponent.
    def talk(self) -> Text:
        message = None
        if not self._conversation_reaches_the_end():
            utterance_text, utterance_action = generate_response(
                history=self.conversation_history[-1],
                profile=self.profile,
                event_string=self.get_current_event_string(),
                topic=self.current_topic,
                goal=self.goal,
            )
            # utterance_text = f"This is a message from {self.name}."
            if utterance_text != "":
                message = {
                    "speaker": self.profile["name"],
                    "utterance": utterance_text,
                    "action": utterance_action,
                }
                self.conversation_history[-1].append(message)
        return message

    def _conversation_reaches_the_end(self):
        return True if len(self.conversation_history[-1]) == 8 else False

    def set_status(self, status: Text):
        self.status = status

    def set_current_event(self, world_time: datetime):
        for index in range(len(self.event_queue)):
            event_time = datetime.strptime(
                f"2024-03-03 {self.event_queue[index].time}", "%Y-%m-%d %H:%M"
            )
            if world_time < event_time:
                if index > 0:
                    self.current_event = self.event_queue[index - 1]
                    break
                else:
                    break
            elif world_time == event_time:
                self.current_event = self.event_queue[index]
            else:
                continue

    def get_current_event_string(self) -> Text:
        # event_string = (f"""Time: {self.current_event.time}\n"""
        #                 f"""Event detail: {self.current_event.description}\n"""
        #                 f"""{self.profile['name']} feels {self.current_event.emotion_labels} and that mood will affect {self.profile['name']} for {self.current_event.effective_duration}.""")
        event_string = f"{self.current_event.name};{self.current_event.emotion_labels};{self.current_event.description}"
        return event_string
