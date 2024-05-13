from scipy.interpolate import BarycentricInterpolator as BI
import matplotlib.pyplot as plt
import numpy as np

class NavigationTrajectory:

    def __init__(self):
        print("Created navigation trajectory object")

# Trajectory generator starts here    
    def Lmove(self,Is,Gs):
        
        
        
        Ix = Is[0]
        Iy = Is[1]
        Gx = Gs[0]
        Gy = Gs[1]

        # Linear move
        xil = np.array([Ix,  Gx ]) #2 , 5.5,
        yil = np.array([Iy,  Gy]) # 8, 7,
        pl  = BI(xil,yil)

        x  = np.linspace(Ix,Gx,100)
        y  = pl(x)

        return x,y



    def Pmove(self,Is,Gs,P1):
        
        Ix  = Is[0]
        Iy  = Is[1]
        Gx  = Gs[0]
        Gy  = Gs[1]
        P1x = P1[0]
        P1y = P1[1]

        # Parabolic move
        xip = np.array([Ix,P1x, Gx ])
        yip = np.array([Iy,P1y, Gy])
        pp  = BI(xip,yip)

        x  = np.linspace(Ix,Gx,100)
        y  = pp(x)

        return x,y


    def Cmove(self,Is,Gs,P1,P2):
        '''
        This function generates trajectory by performing interpolation
        '''
        
        Ix  = Is[0]
        Iy  = Is[1]
        Gx  = Gs[0]
        Gy  = Gs[1]
        P1x = P1[0]
        P1y = P1[1]
        P2x = P2[0]
        P2y = P2[1]

        # Cubic move
        xic = np.array([Ix,P1x,P2x, Gx ])
        yic = np.array([Iy,P1y,P2y, Gy])
        pc  = BI(xic,yic)
        
        

        x  = np.linspace(Ix,Gx,100)
        y  = pc(x)
        
        yint = np.interp(x,xic,yic)

        return x,yint



    def getEquidistantPoints(self,p1, p2, parts):
        
        x = np.linspace(p1[0],p2[0],parts+1)
        y = np.linspace(p1[1], p2[1], parts+1)
        
        return np.array([x,y])

    def Cmove2(self,Is,P1,P2,Gs,parts):
        '''
        This function generates trajectory by performing equipoint distance
        '''
        
        Ix  = Is[0]
        Iy  = Is[1]
        
        P1x = P1[0]
        P1y = P1[1]
        
        P2x = P2[0]
        P2y = P2[1]
        
        Gx  = Gs[0]
        Gy  = Gs[1]
        
        c1 = self.getEquidistantPoints(Is,P1,parts) # path from inital pos to middle state 1
        
        c2 = self.getEquidistantPoints(P1,P2,parts) # path from mid state 1 to mid state 2
        
        c3 = self.getEquidistantPoints(P2,Gs,parts) # path from mid state 2 to goal

        return np.concatenate( (c1,c2,c3),axis=1)
    





    def Gen_MoveC2(self,rawinp,rawtarget,i,parts=100):
        '''
        

        Parameters
        ----------
        rawinp : numpy array
            input array of 6 inputs: VIX1,VIX2,VGX1,VGX2,OIX1,OIX2
        rawtarget : numpy array
            VM1X1,VM2X2,VM2X1,VM2X2.
        i : int
            observation index.
        parts : int, optional
            number of points in a trjactory.There will be 3 additional points. The default is 100.

        Returns
        -------
        Trajectory array.

        '''
        #i=100
        I  = np.array([rawinp[i,0],rawinp[i,1]])
        M1 = np.array([rawtarget[i,0],rawtarget[i,1]])
        M2 = np.array([rawtarget[i,2],rawtarget[i,3]])
        G  = np.array([rawinp[i,2],rawinp[i,3]])
        testmove = self.Cmove2(I,M1,M2,G,parts)
        
        return testmove

    def Generate_All_MoveC2(self,rawinp,rawtarget,parts=100):
        '''
        

        Parameters
        ----------
        rawinp : numpy array
            input array of 6 inputs: VIX1,VIX2,VGX1,VGX2,OIX1,OIX2
        rawtarget : numpy array
            VM1X1,VM2X2,VM2X1,VM2X2.
        parts : int, optional
            number of points in a trjactory.There will be 3 additional points. The default is 100.

        Returns
        -------
        All trajectory array.

        '''
        observation = len(rawinp)
        trajectory = []
        
        k=0
        
        while(k<observation):
            trajectory.append(self.Gen_MoveC2(rawinp,rawtarget,k,parts))
            k=k+1
            
        
        
        return np.asarray(trajectory)
    
    def CmoveEqDist(self,Is,P1,P2,Gs, parts = 100):
        '''
        This function generates trajectory by performing equipoint distance
        Inputs:
            IS: Inital state x and y coordinates
            P1: Mid state 1  x and y coordinates
            P2: Mid state 2  x and y coordinates
            Gs: Goal state   x and y coordinates
        
        Return:

            Trajectory of specifeid parts/points

        '''
        
        c1 = self.getEquidistantPoints(Is,P1,parts) # path from inital pos to middle state 1
        
        c2 = self.getEquidistantPoints(P1,P2,parts) # path from mid state 1 to mid state 2
        
        c3 = self.getEquidistantPoints(P2,Gs,parts) # path from mid state 2 to goal

        Trajectory = np.concatenate( (c1,c2,c3),axis=1)
        return Trajectory

    def InputToTrajectory(self,inp_vec, targ_vec, parts=100):

        '''
            Wrapper to CmoveEqDist

            Inputs:
                inp_vec: A single vector/array with 6 inputs. x,y coordinates of inital state and goal state
                targ_vec: A single vector/array with 4 inputs. x,y coordinates of two middle states
            Return:
                Trajectory of specifeid parts/points
        '''

        IS = inp_vec[0:2] # x and y for initial state
        GS = inp_vec[2:4] # x and y for goal state

        P1 = targ_vec[0:2] # x and y for middle state 1
        P2 = targ_vec[2:4] # x and y for middle state 2

        Trajectory = self.CmoveEqDist(IS,P1,P2,GS, parts)

        return Trajectory



    


    # ---------------------------------------Compare methods starts here-----------------------------------------------------

    def IsClose(self,A,B,th):

        C = [A[0]-B[0],A[1]-B[1]]
        dist = np.linalg.norm(C)
        if(abs(dist)<(th) ):
            return True
        else:
            return False

    def CheckBad(self,Vmove,Omove,th):
        
        if(len(Vmove)>=len(Omove)):
            observation = len(Omove)
        else:
            observation = len(Vmove)


        pt = len(Vmove[0][0]) # number of points in a traj
        m = 0
        n = 0
        count = 0
        index = np.ones([observation])
        while(m<observation):


            while(n<pt):
                V_ppoints = Vmove[m,:,n]
                O_points  = Omove[m,:,n]

                if(self.IsClose(V_ppoints,O_points,th) or self.IsOutRange(V_ppoints) or self.IsNegative(V_ppoints)):
                    count = count+1
                    n=pt # get outta loop
                    index[m]=0

                else:
                    count = count
                    n=n+1
        n=0
        m=m+1

        return count,index

    def GetLabel(self,Vmove,Omove):

        count,index = self.CheckBad(Vmove,Omove)
        return index

    def Bad_Counter(self,Vmove,Omove,thld):
        
        bad_count = np.zeros(len(thld))
        i=0
        while(i<len(bad_count)):
            bad_count[i] = self.CheckBad(Vmove,Omove,thld[i])[0]
            i=i+1

        return bad_count

    def IsOutRange(self,A):

        if(A[0]>1 or A[1]>1):
            return True
        else:
            return False

    def IsNegative(self,A):

        if(A[0]<0 or A[1]<0):
            return True
        else:
            return False




    def checkarray(self,move):

        '''
        input: move trajectory

        Return

        True: if theres a point in the move-trajectory greater than 1 or less than 0

        False: all the values are within 0-1 range

        '''
        length = len(move)

        i=0

        while(i<length):
            if( move[i]<0 or abs(move[i])>1 ):

                i = length
            return True

        else:
            i=i+1

        return False

    def CheckBadRange(self,array):
        '''
        input: a set of generated trajectories

        output: number and index of trajectories out of range (<0 or >1)
        '''

        Observation = array.shape[0]

        moveX = np.zeros(array.shape[2])
        moveY = np.zeros(array.shape[2])

        i=0

        count = 0
        index = []

        while(i<Observation):

            moveX = array[i,0,:]  # X coordinates of trajectory
            moveY = array[i,1,:]  # Y coordinates of trajectory

            if (self.checkarray(moveX) or self.checkarray(moveY)):
                count = count+1
                index.append(i)
                i=i+1

            else:

                i=i+1

        return count, index


    def StaticCheckBad(self,Vmove,Opos,th):

        """
        Return an array of integers.

        :param kind: Optional "kind" of ingredients.
        :raise: If the kind is invalid.
        :return: Bad move counts, Array of indices for bad trajectories.
        :rtype: int , Array[int]

        """


        # observation = len(Vmove)
        pt = len(Vmove[0]) # number of points in a traj
        # m = 0
        n = 0
        count = 0
        #index = []
        #while(m<observation):
            
        while(n<pt):
            V_ppoints = Vmove[:,n]
            O_points  = Opos

            if(self.IsClose(V_ppoints,O_points,th) or self.IsOutRange(V_ppoints) or self.IsNegative(V_ppoints)):
                
                count = count+1
                n=pt # get outta loop
                #index.append(m)
            else:
                
                count = count
                n=n+1
            
        n=0
        #print(m)
        #m=m+1
        
        return count
    

    # plots

    def HitCheckPlot(self,obstacle_x,obstacle_y,threshold,trajectory):
        maxlen = len(trajectory[0])-1
        circle1 = plt.Circle((obstacle_x, obstacle_y), threshold, color='r')
        plt.figure(figsize=(12,12))
        plt.gca().add_patch(circle1)
        plt.plot(trajectory[0],trajectory[1],'*',clip_on=False)
        plt.plot(trajectory[0,0],trajectory[1,0],marker = 'o', ms = 20, mec = 'g') # mark starting position
        plt.plot(trajectory[0,maxlen],trajectory[1,maxlen],marker = 'X', ms = 20, mec = 'g') # mark end/goal position
        plt.plot(obstacle_x,obstacle_y, marker = 'X', ms = 20, mec = 'g') # mark obstacle position
        plt.xlim([0,1])
        plt.ylim([0,1])