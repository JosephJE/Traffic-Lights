import numpy as np


class Container:

    def __init__(self, frame_id = 0, frame_path = "", traffic_lights = [], auxilary = None, locations = None):
        self.frame_id = frame_id
        self.frame_path =frame_path
        self.traffic_lights = traffic_lights
        self.auxilary = auxilary
        self.locations = locations
        self.prev_container = None