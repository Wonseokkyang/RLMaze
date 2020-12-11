# Reinforced Learning Maze
Author: Won Seok Yang

GIT of the agent exploring the maze for the first time. <br/>
![alt text](https://github.com/Wonseokkyang/RLMaze/blob/master/maze_results/5x5/5x5_n%3D0_i%3D0.gif?raw=true)

GIF of agent intelligence after solving the 5x5 maze 100 times. <br/>
![alt text](https://github.com/Wonseokkyang/RLMaze/blob/master/maze_results/5x5/5x5_n%3D0_i%3D100.gif?raw=true)
```
alpha:              0.9
gamma:              0.9
epsilon:            0.9
```

First attempt in creating a machine learning agent using reinforced learning and first standalone project using python. The agent starts with no knowlege of the maze environment and relys on new experiences to learn the best action to take in any given state.

From this project I learned many new things including the importance of correctly defining markov states, dataFrames, reward function application and much more.

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

### N-Step look ahead:
After getting the reinforced learning agent implemented, the next step was n-step lookahead. The agent would no longer have to learn by taking an action but will be able to simulate n-steps ahead of it's current given state. <br/>
This is achieved by setting a small red circle as a 'flag' to indicate the agents current position and simulating possible moves from the agents current position ```n``` moves/action deep. <br/>

![alt text](https://github.com/Wonseokkyang/RLMaze/blob/master/maze_results/5x5/5x5_n%3D2.gif?raw=true) <br/>
Although n-step look ahead is more effective in the agent taking an action for the depth of n, the run time of the program becomes longer because the agent looks ahead at every state transition. The program runtime of the agent using 3-step look ahead is 4x times longer than the runtime of an agent using the regular 0-step learning algorithm. <br/>
There are a few ways to optimize this process: <br/>
- We can trim off early:
  - The paths where the agent receives a heavy penalty, such as walls and out of bounds.
  - If the change in q-value of the lookahead simulation is under a certain threshold.
  - Track the visited cells so the agent doesn't simulate paths/cells it has previously visited/calculated. <br/>

With these learnings, I moved on to the next stage of learning reinforced learning- reinforced learning applied to two agents simultaneously in the form of a [Tic-Tac-Toe](https://github.com/Wonseokkyang/RLTicTacToe) game.