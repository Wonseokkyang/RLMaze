# Reinforced Learning Maze
Author: Won Seok Yang

First attempt in creating a machine learning agent using reinforced learning and first standalone project using python. The agent starts with no knowlege of the maze environment and relys on new experiences to learn the best action to take in any given state.

### libraries used:
graphics - for graphical display of agent and maze environment <br/>
```pip install graphics```

matplotlib - for analyzing performance differences in changing agent values <br/>
```pip install matplotlib```

pandas & numpy - used for dataFrame storage and array search/manipulation in agent's value table <br/>
```pip intall numpy``` <br/>
```pip install pandas``` <br/>

## Key values:
Agent values: <br/>
> alpha - How heavily the learning algorithm gets changed towards a reward/penalty <br/>
> gamma - The discount factor of future rewards (0 nearsighted, 1 farsighted) <br/>
> epsilon - the weight of the algorithm's greediness. A higher epsilon value causes agent to prefer choosing actions that will lead to a reward while a lower epsilon value encourages the agent to explore new actions. <br/>

Environment values: <br/>
```
OUT_OF_FRAME = -100.0   #out of bounds penalty
WALL = -10.0            #hitting a wall penalty
MOVE = -0.1             #cost for moving into a square
TARGET = 100.0          #finding the end of the maze reward
```

### Initial findings:
The first agent was tested in a 5x5 (11x11 including walls) maze environment where the agent would start on the opposite end of the maze exit. This initial test was to see the relationship between the agent's alpha value and agent performance. 
The two line graphics show the number of steps/actions the agent took before reaching maze completion (y-axis) and the iteration number of the agent (x-axis). One interation is starts the agent, represented by the pink circle, at it's initial position (top left) and the iteration is concluded when the agent reaches it's goal, the green square. <br>
Although the agent with an alpha value of 0.2 takes more than a thousand additional steps/actions to reach the exit, when compared side by side with the agent with an alpha value of 0.9, they quickly converge.

![alt text](https://github.com/Wonseokkyang/RLMaze/blob/master/maze_results/5x5/together.jpeg?raw=true)
<br/>

The result of different alpha values in the 5x5 maze was not as decisive as I thought it would be so I tested the agent in a larger environment.
![alt text](https://github.com/Wonseokkyang/RLMaze/blob/master/maze_results/11x11/mazeGraphics_11x11.jpeg?raw=true)
![alt text](https://github.com/Wonseokkyang/RLMaze/blob/master/maze_results/11x11/Comparison_of_different_Alpha_values.png?raw=true) <br/>
This environment is 4x larger and provided a clearer picture for the relationship bewteen alpha value and performance. With the larger environment, we can clearly see that the alpha value of 0.9 preformed better than a value of 0.2.

