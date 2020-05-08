import numpy as np

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
    return np.array(out) #111 int + speed

def addDirection(NewState,OldState):
    if OldState == None:
        dir = np.zeros(2)
    else:
        dir = NewState[-2:]-OldState[-2:]
    return NewState.concatenate(dir)