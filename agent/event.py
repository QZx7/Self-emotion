class Event:
    def __init__(
        self, name, description, time, emotion_labels, effective_duration
    ) -> None:
        self.name = name
        self.description = description
        self.time = time
        self.emotion_labels = emotion_labels
        self.effective_duration = effective_duration

    def get_event_string(self):
        event_string = f"Time: {self.time}\nEvent detail: {self.description}\nThe subject feels {self.emotion_labels} and that mood will affect the subject for {self.effective_duration}."
        return event_string
