import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import namedtuple
"""
def decreaseObservationSpace(observationSpace_old):
    blocks = np.zeros(shape=(6, 18))
    ball = np.zeros(2)
    slider = -1

    i = 0
    j = 0

    # check if the pixels at the konwn block positions are black (block gone) or not (block still there)
    for x in range(57, 88, 6):
        for y in range(8, 152, 8):
            blocks[i][j] = observationSpace_old[x][y][0] != 0
            j += 1
        i += 1
        j = 0

    # get the ball position by finding a pixel with the ball color and checking if either left and right are other colors or below and above it
    # this should always be the case for the ball, since there is no corner in the game where on both sides you have the ball color
    ballFound = False
    for x in range(209, 32, -1):
        if ballFound: break
        for y in range(8, 151):
            if ballFound: break
            if np.array_equal(observationSpace_old[x][y][:], np.array([200, 72, 72])):
                if (not np.array_equal(observationSpace_old[x][y - 1][:], np.array([200, 72, 72])) and np.array_equal(observationSpace_old[x][y + 1][:], np.array([200, 72, 72])) and not np.array_equal(observationSpace_old[x][y + 2][:], np.array([200, 72, 72]))) or (not np.array_equal(observationSpace_old[x + 1][y][:], np.array([200, 72, 72])) and np.array_equal(observationSpace_old[x - 1][y][:], np.array([200, 72, 72])) and not np.array_equal(observationSpace_old[x - 2][y][:], np.array([200, 72, 72]))):
                    ball = np.array([x - 1, y + 1])
                    ballFound = True

    # get the slider position
    sliderFound = False
    for y in range(8, 151):
        if sliderFound: break
        if np.array_equal(observationSpace_old[190][y][:], np.array([200, 72, 72])):
            b = True
            for k in range(1, 15):
                b = b and np.array_equal(observationSpace_old[190][y + i][:], np.array([200, 72, 72]))
            if b:
                slider = y + 8
                sliderFound = True
       
    out= blocks.flatten().astype(int).tolist()
    out.append(slider)            
    out= out + ball.tolist()
    return np.array(out,dtype=int) #111 int + speed
"""


def decreaseObservationSpace(observation):
    observation = np.sum(observation, axis=2)
    blocks = observation[57:88:6,8:152:8].flatten() #try to avoid flatten and use ravel for performance
    blocks = np.where(blocks > 0, 1, -1).astype(float) # try ti avoid this for performance
    

    sliderLine = observation[190,8:152]
    slider = float(np.nonzero(sliderLine)[0][0] - 72)

    
    ballSpace = observation[93:189,8:152]
    ball=np.nonzero(ballSpace)
    if(ball[0].shape[0]>0):
        ballx=ball[0][0] - 48 #normalizing
        bally=ball[1][0] - 72 
    else:
        ballx = 0
        bally = 0

    out = np.zeros(108+1+2)
    out[0:108]=blocks
    out[108]=slider
    out[109]=float(ballx)
    out[110]=float(bally)
    return out
    
    
def addDirection(NewState,OldState=None):
    l1 = list(NewState)
    
    if OldState is None:
        dir = np.zeros(2,dtype=float)
    else:
        dir = NewState[-2:]-OldState[-2:]
    l2 = list(dir)
    l1=l1+l2
    return np.array(l1)
    
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

def extract_tensors(experiences, device):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    
    # convert into tuple of tensors instead of tuple of ndarrays
    bs = [torch.from_numpy(a).to(device) for a in batch.state]
    bns = [torch.from_numpy(a).to(device) for a in batch.next_state]

    t1 = torch.cat(bs)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(bns)
    
    return (t1,t2,t3,t4)

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    durations_t = torch.tensor(values, dtype=torch.float)
    plt.plot(durations_t.numpy())
    
    plt.pause(0.001)
