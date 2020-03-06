from TrafficL_Manager import TrafficLightMan


class controller:
    def __init__(self):
        self.myTFmanager = TrafficLightMan()

    def run(self, buffer_path):
        file = open(buffer_path, "r")
        for line in file:
            self.myTFmanager.execute_frame(line)
            self.myRoadmanager.execute_frame(my)

