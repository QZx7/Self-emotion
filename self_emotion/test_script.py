from datetime import datetime

if __name__ == "__main__":
    world_time = datetime.strptime("14:00", "%H:%M")
    # hour = datetime.strftime(world_time, "%H:%M")

    hour = datetime.strptime("16:00", "%H:%M")

    print(world_time < hour)
