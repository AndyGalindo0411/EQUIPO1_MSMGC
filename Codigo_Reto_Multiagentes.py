from mesa import Agent, Model
from mesa.space import MultiGrid, SingleGrid
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector
from matplotlib.colors import ListedColormap

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["animation.html"] = 'jshtml'
matplotlib.rcParams['animation.embed_limit'] = 2**128

import numpy as np
import pandas as pd
import time
import datetime

def get_grid(model):
    grid = np.full((model.grid.width, model.grid.height), 0)

    for cell in model.grid.coord_iter():
        cell_content, cell_pos = cell
        agent = cell_content[0] if cell_content is not None and len(cell_content) > 0 else None
        if agent is not None:
            if hasattr(agent, 'maiz'):
                grid[cell_pos[0], cell_pos[1]] = agent.maiz
            elif hasattr(agent, 'tractor'):
                grid[cell_pos[0], cell_pos[1]] = agent.tractor

    return grid


##          """
##          agent = cell_content[0]  # Accede al primer agente en la lista
##          print("cell content: ")
##          print(agent)
##          print(agent.dirt)
##          """
##
##          if hasattr(cell_content[0], 'maiz'):
##            agent = cell_content[0]
##            grid[cell_pos[0], cell_pos[1]] = agent.maiz
##
##          if hasattr(cell_content[0], 'tractor'):
##            agent = cell_content[0]
##            grid[cell_pos[0], cell_pos[1]] = agent.tractor
##
##  return grid

class MaizAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.maiz = 1

    def step(self):
        pass

class TractorAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.next_step = None
        self.tractor = 2
        self.number_of_steps = 0

    def move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )

        open_positions = [pos for pos in neighbors if not
                          any(isinstance(agent, TractorAgent)
                              for agent in self.model.grid.get_cell_list_contents(pos))]
        if open_positions:
            self.next_step = open_positions[(np.random.randint(len(open_positions)))]
            self.model.grid.move_agent(self, self.next_step)
            self.number_of_steps += 1

    def harvest(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for agent in cellmates:
            if isinstance(agent, MaizAgent):
                agent.maiz = 0
                self.model.grid.remove_agent(agent)

    def step(self):
        self.move()
        self.harvest()

class HarvestModel(Model):

    def __init__(self, T, M, width, height):
        self.num_tractors = T
        self.num_maiz = M
        self.grid = MultiGrid(width, height, False)
        self.schedule = BaseScheduler(self)
        self.total_maiz = 0

        self.x_coords = [i for i in range (width)]
        self.y_coords = [j for j in range (height)]
        self.coords = [(x, y) for x in self.x_coords for y in self.y_coords if x < width - 1 or y < height - 1] #para que se llene -1 casilla 
        #self.coords = [(x, y) for x in self.x_coords for y in self.y_coords if x < width - 1 and y < height - 1] #para que se llene -1row y una columna 
        self.n = 0
        print(self.coords)

        self.r = 0

        while (self.coords):
            m = MaizAgent(self.r, self)
            self.coord = self.coords.pop(self.n)
            x = self.coord[0]
            y = self.coord[1]
            self.schedule.add(m)
            self.grid.place_agent(m, (x, y))
            self.r += 1
            
        start = self.r + 1
        finish = self.num_tractors + start

        for i in range(start, finish):
            t = TractorAgent(i, self)
            self.schedule.add(t)
            self.grid.place_agent(t, (x, y)) #jala para una casilla pero todo es rojo :| 
            #self.grid.place_agent(t, (x, y + 1)) #solo jala sin row y sin columna
                                                    #si se usa con la casilla jala pero el terreno es blanco y el agente tractor tiene color

        self.datacollector = DataCollector(model_reporters={'Grid': get_grid})

    def maiz_counter(self):
        maiz_left = 0
        for cell in self.grid.coord_iter():
            cell_content, cell_pos = cell
            if len(cell_content) != 0:
                if hasattr(cell_content[0], 'maiz'):
                    agent = cell_content[0]
                    if agent.maiz == 1:
                        maiz_left += 1
        return maiz_left

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.maiz_left_total = self.maiz_counter()

GRID_SIZE = 12
num_tractors = 1
num_dirt = GRID_SIZE^2
num_generations = 500
num_steps = 0

start_time = time.time()
model = HarvestModel(num_tractors, num_dirt, GRID_SIZE, GRID_SIZE)
for i in range(num_generations):
    model.step()
    if (i == (num_generations - 1)):
        for cell in model.grid.coord_iter():
            cell_content, cell_pos = cell
            if len(cell_content) != 0:
                if hasattr(cell_content[0], 'quantity_of_steps'):
                    agent = cell_content[0]
                    num_steps += agent.quantity_of_steps

print(f"total number of steps: {num_steps - num_tractors}")
print(f"Execution time: ", str(datetime.timedelta(seconds = (time.time() - start_time))))


all_grid = model.datacollector.get_model_vars_dataframe()
final_maiz_left = model.maiz_left_total

print(f"total maiz left: {final_maiz_left}")

fig, axis = plt.subplots(figsize=(5, 5))
axis.set_xticks([])
axis.set_yticks([])

colors = [(1, 1, 1),  # Blanco
          (0, 1, 0),  # Rojo (color del agente en movimiento)
          (0.5, 0, 0)]  # Azul (color del agente estacionario)
cmap_custom = ListedColormap(colors)

patch = plt.imshow(all_grid.iloc[0].iloc[0], cmap=cmap_custom)

prev_lines = []

# Crear líneas horizontales y verticales una vez
for x in range(GRID_SIZE + 1):
    for y in range(GRID_SIZE + 1):
        h_line, = axis.plot([-0.5, GRID_SIZE - 0.5], [y - 0.5, y - 0.5], color='gray', linestyle="-", linewidth=0.5)
        v_line, = axis.plot([x - 0.5, x - 0.5], [-0.5, GRID_SIZE - 0.5], color="gray", linestyle="-", linewidth=0.5)
        prev_lines.extend([h_line, v_line])

def animate(i):
    global prev_lines
    for line in prev_lines:
        line.remove()
    prev_lines.clear()

    patch.set_data(all_grid.iloc[i].iloc[0])

    anim_lines = []
    for x in range(GRID_SIZE + 1):
        for y in range(GRID_SIZE + 1):
            h_line, = axis.plot([-0.5, GRID_SIZE - 0.5], [y - 0.5, y - 0.5], color='gray', linestyle="-", linewidth=0.5)
            v_line, = axis.plot([x - 0.5, x - 0.5], [-0.5, GRID_SIZE - 0.5], color="gray", linestyle="-", linewidth=0.5)
            anim_lines.extend([h_line, v_line])

    prev_lines = anim_lines

anim = animation.FuncAnimation(fig, animate, frames=num_generations)
plt.show()
