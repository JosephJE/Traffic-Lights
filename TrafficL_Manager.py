#!/usr/bin/env python
# coding: utf-8

# A Fully-Convolutional Network using VGG weights for the encoder portion trained on the CityScape dataset to distinguish pixels into the following categories: road, sidewalk, pedestrian, vehicle (including bicycles), background.
from TL_Container import Container

from Utills import readImage, AnalyzeImageClassic


class TrafficLightMan:
    def __init__(self):
        self.data = Container
        self.first_frame = True
        self.loaded_frame = None

    def init_container(self, frame_path, id):

        if (self.first_frame):
            self.first_frame = False
        else:
            self.data.prev_container = Container(self.data.frame_id, self.data.frame_path, self.data.traffic_lights,
                                                 self.data.auxilary, self.data.locations)

        self.data.frame_path = frame_path
        self.data.frame_id = id



    def execute_frame(self, frame_path, id):
        self.init_container(frame_path, id)
        self.getCandidates()
        self.confirmCandidates()
        self.calculateDistances()

    def getCandidates(self) :
        frame = readImage(self.data.frame_path)
        self.data.candidates, self.data.auxilary = AnalyzeImageClassic(frame)
                            
    def confirmCandidates(self):
        pass




    def calculateDistances(self):
        pass

# def rgb2gray(rgb):
#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     return gray


# In[100]:


# Show generated data


# In[7]:


# dat_gen = data_generator(batch_size=10, im_size=1000)

# In[8]:


# ims, gts = dat_gen.__next__()


# In[10]:
# path = r"C:\Users\RENT\Desktop\mobileye\CityScapes\leftImg8bit\train\aachen\aachen_000015_000019_leftImg8bit.png"
# g_im = imageio.imread(path)
# visualize_mag(g_im)
# visualize_prediction(ims, gts)

# In[ ]:




