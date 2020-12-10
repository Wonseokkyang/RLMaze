"""
#####################################################
##                  RL Maze Agent                  ##
#####################################################
#   
#   Name: Won Seok Yang
#   
##PRE:  'maze.txt' is a valid maze- 
#           every row has an equal number of elements
#           the maze has one or more solution
#
#   Agent will start the maze on (0,0)top left and try to reach
#   target at (numberOfRows-1, numberOfCols-1)bottom right
#
#   Green rectangle:    walls [reward = ?]
#   Yellow rectangle:    open space [reward = ?]
#   Blue  oval:          agent/explorer
#   Purple rectangle:    target/exit
#
#
#####################################################
##                  Resources                      ##
#####################################################
# Random mazes generated from:      https://www.dcode.fr/maze-generator
# Graphics method reference sheet:  https://mcsp.wartburg.edu/zelle/python/graphics/graphics.pdf
#                                   http://anh.cs.luc.edu/150s07/notes/graphics/graphics_py.htm
# Pandas documentation:             https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
# Matplotlibs documentation:        https://matplotlib.org/tutorials/introductory/pyplot.html
######################################################
"""
#without optimization the comparelookahead() takes a min of 20*cycles+ 20*cycles*20^4^2
from graphics import * 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#for demo only, continues from saved csv file
DEMO = False
SAVE_FILE = False
FILENAME = 'maze5x5_1'
CYCLE_FILE_NUM = 0
CSV_NAME = 'brain_saves/'+FILENAME+'_'+str(CYCLE_FILE_NUM)+'.csv'   
#possibly suppliment the csv files with a txt file with the agent's specs (alpha, gamma, reward values...)

UNIT =  30              #unit size of squares in the graphical representation
SPEED = 0.025            #the speed at which the screen renders (refreshrate) in seconds
CYCLES = 1             #the number of times the agent will travel through the maze till completion

ALPHA = 0.9             #how heavily the learning algorithm gets changed toward a positive reward (learning rate)
GAMMA = 0.9             #the discount factor of future rewards (0 nearsighted vs 1 farsighted)
EPSILON = 0.9           #the weight of the algo's 'greediness', less greedy = explore, more greedy = exploit
LOOK_AHEAD_DEPTH = 2    #0 for greedy algo only, else LOOK_AHEAD_DEPTH = 'n'

#Reward values:
OUT_OF_FRAME = -100.0   #out of bounds penalty
WALL = -10.0            #hitting a wall penalty
MOVE = -0.1             #moving into a square penalty
TARGET = 100.0          #finding the end of the maze reward

ANNOUNCE_AGENT_MOVES = False    #flag to print the agent's moves in terminal
DRAW_MAZE = True               #toggle for graphical representation of the maze

class Maze:
    #Populate self with maze from FILENAME and sets default values for pos&actions
    def __init__(self, mazeTextFile):
        self.mazeList=[]
        mazeText=open('maze/original/'+mazeTextFile+'.txt')
        for line in mazeText:
            rowList=[]
            for ch in line[:-1]:    #[:-1] all but the last char- new line 
                rowList.append(ch)
            self.mazeList.append(rowList)
        mazeText.close()

        #Cardinal directions:
        # 0 = UP,   1 = DOWN,   2 = LEFT,   3 = RIGHT
        self.actions = [0, 1, 2, 3] 
        self.numberOfCols = len(self.mazeList[0])
        self.numberOfRows = len(self.mazeList)
        # print("Number of rows: %d \tNumber of cols: %d" % 
        #             (self.numberOfRows, self.numberOfCols))

        #Agent oval objct init, starting position
        self.pos = (0, 0)   #always start at 0,0
        xpos, ypos = self.pos
        self.mazeTextOut()

        #Graphics window init
        if DRAW_MAZE == True:
            self.win = GraphWin("Maze Visual "+FILENAME, 
                width=UNIT*self.numberOfCols, 
                height=UNIT*self.numberOfRows)    #setup display window according to maze size
            self.agent = Oval( Point(xpos*UNIT, ypos*UNIT), Point(xpos*UNIT+UNIT, ypos*UNIT+UNIT))
            self.agent.setFill("pink")
            self.flag = Oval( Point(xpos*UNIT+UNIT/4, ypos*UNIT+UNIT/4), Point(xpos*UNIT+UNIT-UNIT/4, ypos*UNIT+UNIT-UNIT/4))
            self.flag.setFill("red")

        #Maze target position
        self.tpos = (self.numberOfRows-1, self.numberOfCols-1)  #target pos always in the opposite corner
        # print("Starting pos: %s \tTarget pos: %s" % (self.pos, self.tpos))

    def resetAgent(self):  #resets agent to last known/valid x y position
        #Agent starting pos represented by pink circle- starts top left of graphical screen
        xpos, ypos = self.pos
        self.agent.undraw()
        self.agent = Oval( Point(xpos*UNIT, ypos*UNIT), Point(xpos*UNIT+UNIT, ypos*UNIT+UNIT))
        self.agent.setFill("pink")
        self.agent.draw(self.win)
        time.sleep(SPEED)
    
    def drawFlag(self, state):  #a home flag to denote lookahead state origin
        x, y = state
        self.flag.undraw()
        self.flag = Oval( Point(x*UNIT+UNIT/4, y*UNIT+UNIT/4), Point(x*UNIT+UNIT-UNIT/4, y*UNIT+UNIT-UNIT/4))
        self.flag.setFill("red")
        self.flag.draw(self.win)
        time.sleep(SPEED)

    #restarts the maze like it was initialized, resetting agent's tracked xpos, ypos
    def restart(self):
        self.pos = (0,0)
        if DRAW_MAZE == True: self.resetAgent()

    #Takes int argument for action/direction to move the agent to calculate the reward and new self.pos
    def moveAgent(self, direction_num):
        """ Takes int argument for action/direction to move the agent to calculate the reward and new self.pos
            RETURN: self.pos, reward, done
            <tuple> self.pos after move- same state in the case of wall or out of bounds
            <int>   reward value for moving to that state
            <bool>  done flag when the move results in the agent exiting the maze 
            """
        # print('Print type of direction_num in moveAgent(d_num)', type(direction_num)) #demo false -> int    # # CSV TEST
        if direction_num == 0: dy, dx = 1, 0        #UP
        elif direction_num == 1: dy, dx = -1, 0     #DOWN
        elif direction_num == 2: dy, dx = 0, -1     #LEFT
        elif direction_num == 3: dy, dx = 0, 1      #RIGHT
        
        x,y = self.pos
        tx, ty = self.tpos

        #if the agent attempts to move off screen, remove agent from window before resetting
        #agent is out of bounds so give penalty
        if ((x+dx < 0) or (x+dx > self.numberOfRows-1)) or ((y+dy < 0) or (y+dy > self.numberOfRows-1)):
            if ANNOUNCE_AGENT_MOVES == True: print('Agent went out of bounds')
            if DRAW_MAZE == True:
                self.agent.undraw()
                time.sleep(SPEED)
                self.resetAgent()
            reward = OUT_OF_FRAME
            done = False
        #agent hit a wall, blink agent on wall before resetting
        #wall hit so give penalty
        elif self.mazeList[x+dx][y+dy] == '#':
            if ANNOUNCE_AGENT_MOVES == True: print('Agent hit a wall')
            if DRAW_MAZE == True:
                self.agent.move(dx*UNIT,dy*UNIT)
                time.sleep(SPEED)
                for i in range(2):  #blinking animation
                    self.agent.undraw()
                    time.sleep(SPEED/4)
                    self.agent.draw(self.win)
                    time.sleep(SPEED/4)
                self.resetAgent()
            reward = WALL
            done = False
        #agent found the target so reward and trigger done flag
        elif x+dx == tx and y+dy == ty:
            if ANNOUNCE_AGENT_MOVES == True: print('Agent found the target!')
            if DRAW_MAZE == True:
                self.agent.move(dx*UNIT,dy*UNIT)
                self.pos = (x+dx, y+dy)
                time.sleep(SPEED)
            reward = TARGET
            done = True
        #agent landed on regular tile
        else:
            if ANNOUNCE_AGENT_MOVES == True: print('Agent moved')
            if DRAW_MAZE == True:
                self.agent.move(dx*UNIT,dy*UNIT)
                time.sleep(SPEED)
            self.pos = (x+dx, y+dy)
            reward = MOVE
            done = False
        # print('self.pos, reward, done', self.pos, reward, done) # # CSV TEST
        return self.pos, reward, done

    def mazeTextOut(self):
        fwrite = open("mazeOut.txt", 'w')
        for x in range(self.numberOfRows):      #populating maze squares
            for y in range(self.numberOfCols):
                if self.mazeList[x][y] =='#':   #walls
                    fwrite.write('%s %s\n' % (x, y))
        fwrite.close()

    #Graphic visualization of maze
    def drawMaze(self):
        for x in range(self.numberOfRows):      #populating maze squares
            for y in range(self.numberOfCols):
                dSquare = Rectangle(Point(x*UNIT,y*UNIT), Point(x*UNIT+UNIT,y*UNIT+UNIT))
                if self.mazeList[x][y] =='#':   #walls
                    dSquare.setFill("black")
                    dSquare.draw(self.win)
                else:                           #empty squares
                    dSquare.setFill("light grey")
                    dSquare.draw(self.win)

        #Target square to reach represented by a green square- finishes bottom right of screen
        exitSquare = Rectangle(Point((self.numberOfRows-1)*UNIT, (self.numberOfCols-1)*UNIT), 
                        Point((self.numberOfRows-1)*UNIT+UNIT, (self.numberOfCols-1)*UNIT+UNIT))
        exitSquare.setFill("green")
        exitSquare.draw(self.win)
        self.resetAgent()   #draw agent at self.pos
        time.sleep(SPEED)

class Brain:
    #list of actions provided during first func call
    def __init__(self, actions, given_alpha, GAMMA, EPSILON):
        self.actions = actions
        self.alpha = given_alpha
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        """Given state, choose an action according to EPSILON/greediness
            RETURN: action
            <int> action direction- epsilon percent chance or moving to highest qvalue, 100-epsilon percent chance of moving randomly
            """
        self.state_exist_check(state)   #append to q_table if it doesnt exist already
        if np.random.uniform() < self.epsilon:  #greediness 'roll'- if less, then be greedy
            action_choices = self.q_table.loc[str(state), :]     #a list of directional values from 'state' ex: [0, 0, 0.5, 0]   left has highest value
            #from chooseable actions, pick out of the largest/max
            action = np.random.choice(action_choices[action_choices == np.max(action_choices)].index)
        else:   #otherwise, choose random
            action = np.random.choice(self.actions)
        return int(action)

    #updating q_table values
    def calculate(self, state, action, reward, new_state, target):
        """ Takes arguments and updates q-value table according to bellman equation that's influenced by alpha or learn rate
            RETURNS: q_
            <int> q_ is the predicted qvalue for the given state action pair
        """ 
        self.state_exist_check(new_state)   #append the new_state to q_table to add/manip calculations
        q = self.q_table.loc[str(state), action]   #Q value
        # print('After q assignment') # # CSV TEST

        if new_state != target:   #agent didnt find exit
            q_ = reward + self.gamma * self.q_table.loc[str(new_state),:].max()    #max possible Q of new state
        else:   # agent found exit so there is no future state, just give reward
            q_ = reward
        self.q_table.loc[str(state),action] += self.alpha*(q_ - q) #update q_table with difference between estimate and actual * learning rate

    #check if state exists in q_table, if not append it
    def state_exist_check(self, state):
        if str(state) not in self.q_table.index:
            self.q_table = self.q_table.append(     #required to assign a copy of the append q_table to original to keep values for some reason
                pd.Series(      #make an entrie to the q_table according to this format for easy manipulation later
                    [0] * len(self.actions),
                    index = self.q_table.columns,   #columns of series entry according to dataframe(q_table) columns
                    name = str(state),   #the name(left most column for indexing)
                )
            )

    #recursive call to update q_value of all directions from state
    def n_step(self, env, state, depth):
        state = env.pos
        if depth > 0:
            self.state_exist_check(state)
            for action in self.actions: #for every direction move and calculate
                # print('check action:', action, ' state:', state)
                env.pos = state #reset for each direction
                new_state, reward, done = env.moveAgent(action)
                self.calculate(state, action, reward, new_state, env.tpos)
                self.n_step(env, new_state, depth-1)
                # print('after check action:', action, ' state:', state)
        env.pos = state

######################
#   PROGRAM CYCLE    #
######################
def RL_program(given_alpha=ALPHA, n_depth=LOOK_AHEAD_DEPTH):
    print('Alpha =',given_alpha)
    myMaze = Maze(FILENAME)     #init maze
    if DRAW_MAZE == True: myMaze.drawMaze()
    myAgent = Brain(list(range(len(myMaze.actions))), given_alpha, GAMMA, EPSILON)    #init agent brain

    if DEMO == True:    #download the brain to myAgent
        myAgent.q_table = pd.read_csv(CSV_NAME, index_col=0)    #this required changing the return types of some of my function to get working

    cycle_numbers = []
    cycle_count = 0
    agentHistory = []
    for _ in range(CYCLES):
        steps_to_exit = 0   #number of steps it took for the agent to find the exit during this cycle
        reward_sum = 0      #cycle's reward sum
        myMaze.restart()    #reset the maze to starting values, but leaves myAgent's q_table/brain for next cycle
        while True: #this loops until the maze sends the 'done' flag
            steps_to_exit += 1
            #to process, the agent needs it's relation in the maze, s = current agent state
            state = myMaze.pos
            
            #this is for bradley's graphics rendering
            agentHistory.append(state)

            # print('\nchecking q_table before:\n', myAgent.q_table)

            if LOOK_AHEAD_DEPTH > 0:
                if DRAW_MAZE == True: myMaze.drawFlag(state)
                # print('+Calling n_step with state:', state, ' and depth:', LOOK_AHEAD_DEPTH)
                myAgent.n_step(myMaze, state, n_depth)
                myMaze.pos = state
                if DRAW_MAZE == True: myMaze.resetAgent()
                if DRAW_MAZE == True:myMaze.flag.undraw()
            
            # print('checking q_table after:\n', myAgent.q_table)
            # time.sleep(1)
            #choose an action given agents current state using just the q-value table
            action = myAgent.choose_action(state) 


            # #action is an int. need to pass that int to moveAgent to get s', reward and done flag
            new_state, reward, done = myMaze.moveAgent(int(action))
            reward_sum += reward
    
            # print('new_state, reward, done', new_state, reward, done)   # # CSV TEST

            if DEMO == False: 
                #function where the learning takes place- population of q_table
                myAgent.calculate(state, action, reward, new_state, myMaze.tpos)
            else: 
                myAgent.calculate(state, str(action), reward, new_state, myMaze.tpos)
            #a stopgap to a problem that came up with importing exporting csv q_table

            if done == True:    #triggered by maze.moveAgent function
                if SAVE_FILE == True:
                    if (DEMO == False and cycle_count % 10 == 0):   #if you're not running a demo, gather and save q_values for evaluation
                        csv_name = 'brain_saves/'+FILENAME+'_'+str(cycle_count)+'.csv'
                        myAgent.q_table.to_csv(csv_name, index=True, header=True)
                        print('Writing csv file to ',csv_name)
                    print('Cycle number: %s \tSteps taken: %s \tTotal reward: %s' % (cycle_count, steps_to_exit, reward_sum))
                cycle_count += 1
                cycle_numbers.append(steps_to_exit)
                # print(myAgent.q_table)
                print('cycle_count:', cycle_count)
                break;      #break nesting while True loop1
    
    print(agentHistory)
    agentMovetoTxt(agentHistory)
    return cycle_numbers

def agentMovetoTxt(moveList):
    fwrite = open("agentHistory.txt", 'w')
    for pos in moveList:
        x,y = pos
        fwrite.write('%s %s\n' % (x, y))
    fwrite.close()


#################
#   PLOTTING    #
#################
#Return: a list of step averages at cycle index
def getAvgPlotData(number_of_runs, alpha_to_test=ALPHA, depth_to_test=LOOK_AHEAD_DEPTH):
    sum_step_avg_at_cycle_index = [0]*CYCLES    #arr of 0's w/ size=number of CYCLES
    for count_number in range(number_of_runs): #run the program and add to list of cycle
        print('')
        cycle = RL_program(alpha_to_test, depth_to_test)
        for cycle_index in range(CYCLES):
            sum_step_avg_at_cycle_index[cycle_index] += cycle[cycle_index]
    for element in range(len(sum_step_avg_at_cycle_index)): #avg each element of sum list
        sum_step_avg_at_cycle_index[element] = sum_step_avg_at_cycle_index[element]/number_of_runs
    return(sum_step_avg_at_cycle_index)

def singleRunPlot(alpha_to_test, depth_to_test=LOOK_AHEAD_DEPTH):
    plot1 = plt.figure('Alpha =%s Depth=%s' % (alpha_to_test, depth_to_test))
    tic = time.perf_counter()
    plt.plot (RL_program(alpha_to_test, depth_to_test))
    toc = time.perf_counter()
    plt.ylabel('Steps taken')
    plt.xlabel('Cycle number')
    plt.title('alpha=%s depth=%s runtime=%.2f secs' % (alpha_to_test, depth_to_test, (toc-tic)))

def comparisonAlpha():   
    # Running program once with low alpha and showing graph
    print('\nRunning single plot with low alpha')
    low_alpha = 0.2
    singleRunPlot(low_alpha)

    # Running program once with high alpha and showing graph
    print('\nRunning single plot with high alpha')
    high_alpha = 0.9
    singleRunPlot(high_alpha)

    number_of_runs=5
    print('\nGathering average values of low alpha')
    low_alpha_avg = getAvgPlotData(number_of_runs, low_alpha)      #low alpha
    print('\nGathering average values of high alpha')
    high_alpha_avg = getAvgPlotData(number_of_runs, high_alpha)      #high alpha

    # Third diagram with avg runs of high and low for comparison
    plot3 = plt.figure('\nComparison of low vs high Alpha values')
    N = len(low_alpha_avg)

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, low_alpha_avg, width, label='A='+str(low_alpha))
    plt.bar(ind+width, high_alpha_avg, width, label='A='+str(high_alpha))

    plt.ylabel('Avg steps')
    plt.xlabel('Cycle number')
    plt.title('Average comparison over %s runs' % number_of_runs) #shown at top
    plt.legend(loc='best')  #let the lib decide where to put the legend

    print('\nLower alpha minimum solution of %s steps occured at cycle' % min(low_alpha_avg), low_alpha_avg.index(min(low_alpha_avg)))
    print('Higher alpha minimum solution of %s steps occured at cycle' % min(high_alpha_avg), high_alpha_avg.index(min(high_alpha_avg)))
    
    print('\nAll cycles complete!')
    plt.show()
    plt.close()

def compareLookahead():
    low_depth = 0
    high_depth = 3

    singleRunPlot(ALPHA, low_depth)
    singleRunPlot(ALPHA, high_depth)

    number_of_runs=20
    low_tic = time.perf_counter()
    low_depth_avg = getAvgPlotData(number_of_runs, depth_to_test=low_depth)      #low depth
    low_toc = time.perf_counter()
    high_tic = time.perf_counter()
    high_depth_avg = getAvgPlotData(number_of_runs, depth_to_test=high_depth)      #high depth
    high_toc = time.perf_counter()

    plot3 = plt.figure('\nComparison of low vs high depth')
    N = len(low_depth_avg)

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, low_depth_avg, width, label='Depth='+str(low_depth)+' Runtime='+str('{:.2f}'.format(low_toc-low_tic)))   #legend
    plt.bar(ind+width, high_depth_avg, width, label='Depth='+str(high_depth)+' Runtime='+str('{:.2f}'.format(high_toc-high_tic)))   #legend

    plt.ylabel('Avg steps')
    plt.xlabel('Cycle number')
    plt.title('Average comparison over %s runs' % number_of_runs) #shown at top
    plt.legend(loc='best')  #let the lib decide where to put the legend
    print('low:', low_depth_avg)
    print('high:', high_depth_avg)
    print('\nLower depth minimum solution of %s steps occured at cycle' % min(low_depth_avg), low_depth_avg.index(min(low_depth_avg)))
    print('Higher depth minimum solution of %s steps occured at cycle' % min(high_depth_avg), high_depth_avg.index(min(high_depth_avg)))
    
    print('\nAll cycles complete!')
    plt.show()
    plt.close()

def multi_test():
    t_default = threading.Thread( target=RL_program, args=(0.9, 0))
    t_lookahead = threading.Thread( target=RL_program, args=(0.9, 2))

    t_default.start()
    t_lookahead()

    t_default.join()
    t_lookahead.join()

    print('Completed both threads')
    
    



print(RL_program(0.9), 2)
# comparisonAlpha()

# compareLookahead()
# multi_test()
