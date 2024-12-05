import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import build_a_brain as bb

# Use front end user friendly approach
bb.build_network_interface()

# Use back end approach
# root = -1 and progress = -1 when using this approach
net, all_spikes, all_voltages = bb.run_simulation(root=-1, progress=-1, num_steps=1000,
                                                    layer_1_size=100,
                                                    layer_2_size=100,
                                                    layer_3_size=100,
                                                    layer_4_size=100,
                                                    layer_5_size=100,
                                                    connectivity_matrix = np.array([[0,0,0.02,0.02,0.01],
                                                                                    [0.001,0.001,0,0.001,0.01],
                                                                                    [0.001,0.001,0.01,0,0.001],
                                                                                    [0.01,0.01,0.01,0,0.01],
                                                                                    [0.001,0.001,0.01,0.001,0]]),
                                                                                    driving_layer = 3,
                                                                                    driving_neuron_nums = 50)