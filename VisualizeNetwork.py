
# Will Booker
# Based on previous work by Gavin Weiguang Ding

#------------------------------------------------------------------------------#
# Imports
#------------------------------------------------------------------------------#

#Standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()

#Matplotlib imports
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

#Defining colors
White, Light, Medium = 1., 0.7, 0.5
Dark, Darker, Black = 0.3, 0.15, 0.

#------------------------------------------------------------------------------#
# Helper Methods
#------------------------------------------------------------------------------#

#Network grapher class
class NetworkGrapher:

    #Initializes the network grapher
    def __init__(self, layer_width=100):

        #Initializing plot locations
        self.LAYER_WIDTH = layer_width
        self.top_left = [0,0]

        #Initializing values
        self.patches, self.colors = [], []
        self.fig, self.ax = plt.subplots(figsize=(12,6))
        self.layers = []


    #Adds a label to the graph
    def _add_text(self, xy, text, xy_off=[0, 8]):
        plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
                family='sans-serif', size=8)


    #Adds a network layer to the graph
    def add_layer(self, size, num, label=None,
            max_num=16, num_dots=4, offset=[6, -6]):

        #Getting the length of the layer
        layer_length = min(num, max_num)

        #Adding the new layer to the list
        self.layers.append({'length': layer_length, 'offset': offset,
                'size': size})

        #Adding label
        if label == None: label = 'Layer ' + str(len(self.layers))
        self._add_text(self.top_left, label)

        #Adds a rectangle
        top_left = np.array(self.top_left)
        offset = np.array(offset)
        loc_start = top_left - np.array([0, size[0]])
        self.top_left[0] += self.LAYER_WIDTH

        #Setting the omission guidelines
        start_omit = (layer_length - num_dots) // 2 - 1
        end_omit = layer_length - start_omit - 1

        #Creating images
        for index in range(layer_length):

            #Choosing omission
            omit = (num > max_num) and (start_omit < index < end_omit)

            #Appending the image
            if omit:
                position = loc_start + index * offset + np.array(size) / 2
                image = Circle(position, 0.5)
            else:
                position = loc_start + index * offset
                image = Rectangle(position, size[1], size[0])
            self.patches.append(image)

            #Appending the colors
            if omit:        self.colors.append(Black)
            elif index % 2: self.colors.append(Medium)
            else:           self.colors.append(Light)


    #Adds a convolution map between two layers
    def add_conv_mapping(self, start_size=(0,0), end_size=(0,0),
            offset=4, rect_color=Dark):

        #----------------------------------------------------------------------#
        # Previous Layer Rectangle
        #----------------------------------------------------------------------#

        #Getting information about the previous layers
        previous_layer = self.layers[-2]
        next_layer = self.layers[-1]

        #Getting the previous layer offset distance
        dx = (previous_layer['length'] - 1) * previous_layer['offset'][0]
        dy = (previous_layer['length'] - 1) * previous_layer['offset'][1]

        #Getting the start offset
        start_offset = [0, 0]
        start_offset[0] =  previous_layer['size'][0] - offset - start_size[0]
        start_offset[1] = -previous_layer['size'][1] + offset + start_size[1]

        #Getting the start location
        sloc = self.top_left[0] - 2 * self.LAYER_WIDTH, self.top_left[1]
        sloc = sloc[0] + dx + start_offset[0], sloc[1] + dy + start_offset[1]

        #Adding the start patch
        start_patch = Rectangle(sloc, start_size[1], -start_size[0])
        self.patches.append(start_patch)
        self.colors.append(rect_color)

        #----------------------------------------------------------------------#
        # Next Layer Rectangle
        #----------------------------------------------------------------------#

        #Getting the previous layer offset distance
        dx = (next_layer['length'] - 1) * next_layer['offset'][0]
        dy = (next_layer['length'] - 1) * next_layer['offset'][1]

        #Getting the start offset
        end_offset = [0, 0]
        end_offset[0] = dx + offset
        end_offset[1] = dy - offset

        #Getting the start location
        eloc = self.top_left[0] - 1 * self.LAYER_WIDTH, self.top_left[1]
        eloc = eloc[0] + end_offset[0], eloc[1] + end_offset[1]

        #Adding the start patch
        end_patch = Rectangle(eloc, end_size[1], -end_size[0])
        self.patches.append(end_patch)
        self.colors.append(rect_color)

        #----------------------------------------------------------------------#
        # Connecting Lines
        #----------------------------------------------------------------------#

        #Adding line 1
        coords_x = sloc[0], eloc[0]
        coords_y = sloc[1], eloc[1]
        line = Line2D(coords_x, coords_y, lw=0.6)
        self.patches.append(line)
        self.colors.append(Darker)

        #Adding line 2
        coords_x = sloc[0] + start_size[0], eloc[0] + end_size[0]
        coords_y = sloc[1], eloc[1]
        line = Line2D(coords_x, coords_y, lw=0.6)
        self.patches.append(line)
        self.colors.append(Darker)

        #Adding line 3
        coords_x = sloc[0], eloc[0]
        coords_y = sloc[1] - start_size[1], eloc[1] - end_size[1]
        line = Line2D(coords_x, coords_y, lw=0.6)
        self.patches.append(line)
        self.colors.append(Darker)

        #Adding line 4
        coords_x = sloc[0] + start_size[0], eloc[0] + end_size[0]
        coords_y = sloc[1] - start_size[1], eloc[1] - end_size[1]
        line = Line2D(coords_x, coords_y, lw=0.6)
        self.patches.append(line)
        self.colors.append(Darker)


    #Adds a convolution map between two layers
    def add_pool_mapping(self, start_size=(0,0), end_size=(0,0),
            rect_color=Light, offset=3):

        #----------------------------------------------------------------------#
        # Previous Layer Rectangle
        #----------------------------------------------------------------------#

        #Getting information about the previous layers
        previous_layer = self.layers[-2]
        next_layer = self.layers[-1]

        #Getting the previous layer offset distance
        dx = (previous_layer['length'] - 1) * previous_layer['offset'][0]
        dy = (previous_layer['length'] - 1) * previous_layer['offset'][1]

        #Getting the start location
        sloc = self.top_left[0] - 2 * self.LAYER_WIDTH, self.top_left[1]
        sloc = (sloc[0] + dx + previous_layer['size'][0] // 2 -
                start_size[0] // 2, sloc[1] + dy - offset)

        #Adding the start patch
        start_patch = Rectangle(sloc, start_size[1], -start_size[0])
        self.patches.append(start_patch)
        self.colors.append(rect_color)

        #----------------------------------------------------------------------#
        # Next Layer Rectangle
        #----------------------------------------------------------------------#

        #Getting the previous layer offset distance
        dx = (next_layer['length'] - 1) * next_layer['offset'][0]
        dy = (next_layer['length'] - 1) * next_layer['offset'][1]

        #Getting the start location
        eloc = self.top_left[0] - 1 * self.LAYER_WIDTH, self.top_left[1]
        eloc = (eloc[0] + dx + next_layer['size'][0] // 2 - end_size[0] // 2,
                eloc[1] + dy - offset)

        #Adding the start patch
        end_patch = Rectangle(eloc, end_size[1], -end_size[0])
        self.patches.append(end_patch)
        self.colors.append(rect_color)

        #----------------------------------------------------------------------#
        # Connecting Lines
        #----------------------------------------------------------------------#

        #Adding line 1
        coords_x = sloc[0], eloc[0]
        coords_y = sloc[1], eloc[1]
        line = Line2D(coords_x, coords_y, lw=0.6)
        self.patches.append(line)
        self.colors.append(Darker)

        #Adding line 2
        coords_x = sloc[0] + start_size[0], eloc[0] + end_size[0]
        coords_y = sloc[1], eloc[1]
        line = Line2D(coords_x, coords_y, lw=0.6)
        self.patches.append(line)
        self.colors.append(Darker)

        #Adding line 3
        coords_x = sloc[0], eloc[0]
        coords_y = sloc[1] - start_size[1], eloc[1] - end_size[1]
        line = Line2D(coords_x, coords_y, lw=0.6)
        self.patches.append(line)
        self.colors.append(Darker)

        #Adding line 4
        coords_x = sloc[0] + start_size[0], eloc[0] + end_size[0]
        coords_y = sloc[1] - start_size[1], eloc[1] - end_size[1]
        line = Line2D(coords_x, coords_y, lw=0.6)
        self.patches.append(line)
        self.colors.append(Darker)


    #Plots the network
    def plot(self):

        #Iterating over the combined patches and colors
        for patch, color in zip(self.patches, self.colors):

            #Setting the appropriate color
            patch.set_color(color * np.ones(3))

            #Plotting lines
            if isinstance(patch, Line2D):
                self.ax.add_line(patch)

            #Plotting patches
            else:
                patch.set_edgecolor(Black * np.ones(3))
                self.ax.add_patch(patch)

        #Setting plot parameters
        plt.tight_layout()
        plt.axis('equal')
        plt.axis('off')
        plt.show()

        #Saving figure
        self.fig.set_size_inches(8, 2.5)
        fig_dir = './'
        fig_ext = '.png'
        self.fig.savefig(os.path.join(fig_dir, 'NetworkGraph' + fig_ext),
                bbox_inches='tight', pad_inches=0)

#------------------------------------------------------------------------------#
# Main Method
#------------------------------------------------------------------------------#

#Running main method
if __name__ == '__main__':

    #Creating network
    net = NetworkGrapher(layer_width=100)

    #Adding Input Layer
    net.add_layer(size=(80,80), num=2, label='Inputs\n2 @ 500x500')

    #Adding Pooling Layer 1
    net.add_layer(size=(40,40), num=2, label='Pooling x2\n2 @ 250x250')
    net.add_pool_mapping(start_size=(20,20), end_size=(10,10))

    #Adding Convolution Layer 1
    net.add_layer(size=(40,40), num=8, label='Conv 3x3\n8 @ 248x248')
    net.add_conv_mapping(start_size=(6,6), end_size=(2,2))

    #Adding Convolution Layer 2
    net.add_layer(size=(40,40), num=16, label='Conv 3x3\n16 @ 246x246')
    net.add_conv_mapping(start_size=(6,6), end_size=(2,2))

    #Adding Convolution Layer 3
    net.add_layer(size=(40,40), num=32, label='Conv 3x3\n32 @ 244x244')
    net.add_conv_mapping(start_size=(6,6), end_size=(2,2))

    #Adding Pooling Layer 2
    net.add_layer(size=(20,20), num=32, label='Pooling x2\n32 @ 122x122')
    net.add_pool_mapping(start_size=(10,10), end_size=(5,5))

    #Adding Convolution Layer 4
    net.add_layer(size=(20,20), num=16, label='Conv 3x3\n16 @ 120x120')
    net.add_conv_mapping(start_size=(6,6), end_size=(2,2), offset=3)

    #Adding Convolution Layer 5
    net.add_layer(size=(20,20), num=8, label='Conv 3x3\n8 @ 118x118')
    net.add_conv_mapping(start_size=(6,6), end_size=(2,2), offset=3)

    #Adding Convolution Layer 6
    net.add_layer(size=(20,20), num=1, label='Conv 3x3\n1 @ 114x114')
    net.add_conv_mapping(start_size=(6,6), end_size=(2,2), offset=3)

    #Adding Upsampling Layer
    net.add_layer(size=(80,80), num=1, label='Upsampling\n1 @ 500x500')
    net.add_pool_mapping(start_size=(4,4), end_size=(20,20))

    #Adding Smoothing Layer
    net.add_layer(size=(80,80), num=1, label='Smoothing\n1 @ 500x500')
    net.add_conv_mapping(start_size=(20,20), end_size=(4,4), rect_color=White)

    #Adding Smoothing Layer
    net.add_layer(size=(80,80), num=1, label='Discretized\n1 @ 500x500')
    net.add_conv_mapping(start_size=(4,4), end_size=(4,4), rect_color=White)

    #Plotting network
    net.plot()

#------------------------------------------------------------------------------#
#
#------------------------------------------------------------------------------#
