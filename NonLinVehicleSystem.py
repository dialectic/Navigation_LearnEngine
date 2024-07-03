# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:19:49 2023

@author: ai598
"""
import control as ct
import numpy as np


class NonLinVehicleSystem:
    def __init__(self,m,j,c): # constructor

        self.m = m
        self.j = j
        self.c = c

        print("Created Non Linear Vehicle Instance")
    
    # ------------------------------------getters-------------- 
    def get_m(self): 
        return self.m
    
    # getter method 
    def get_j(self): 
        return self.j
    
    # getter method 
    def get_c(self): 
        return self.c  
      
    # ----------------------------------------setters--------------------------
    def set_m(self, m): 
        self.m = m  
    # setter method 
    def set_j(self, j): 
        self.j = j
    # setter method 
    def set_c(self, c): 
        self.c = c    
    
    def updt_wrapper(self):

        """
            Returns vehicle/agent state update function
        """
        
        def myvehicle_updt(t,x,u,params):

            """
                Updates vehicle/agent state variables
            """
            
            m = params.get('m',self.m)
            j = params.get('j',self.j)
            c = params.get('c',self.c)

            # state
            x1= x[0]
            x2= x[1]
            x3= x[2]
            x4= x[3]
            x5= x[4]
            x6= x[5]

            # input
            fx = u[0]
            fy = u[1]
            tz = u[2]



            # dynamical equation
            dx1 = x3
            dx2 = x4
            dx3 = (1/m)*fx*( np.cos(x5*(np.pi/180)) - np.sin(x5*(np.pi/180) )- c*x3 )
            dx4 = (1/m)*fy*( np.sin(x5*(np.pi/180)) + np.cos(x5*(np.pi/180) )- c*x4 )
            dx5 = x6
            dx6 = (1/j)*(tz - c*x6)

            return [dx1,dx2,dx3,dx4,dx5,dx6]
        
        return myvehicle_updt
    
    def output_wrapper(self):

        """
            Returns output update function
        """

        def myvehicle_out(t, x, u, param):
            
            '''
             Uupdates output parameters/variables
            '''
            return x[0:6]
        
        return myvehicle_out


        
    #-----------------------------------------Methods--------------------    

    # Generate the Non-linear system
    def OpenLoopSys(self): 
        
        """
            Returns open loop system instance
        """

        #myvehicle_updt = self.update_func()
        #myvehicle_out = self.Output_func()
        nonlin_myvehicle = ct.NonlinearIOSystem(self.updt_wrapper(), self.output_wrapper(), inputs=['u1','u2','u3'], outputs=['x1','x2','x3','x4','x5','x6'], states=['x1','x2','x3','x4','x5','x6'] )

        

        return nonlin_myvehicle 
    



    def FindEqPoint(self, nonlin_myvehicle, u0 = [0,0,0] , x0 = [1,2,.5,.1,50,.1] ):
    
        """
            return Eq point
            
            Input 1: NonLinearIO system type object. Open Loop System instance
            Input 2: List type. Intial input
            Input 3: List type. Initial position
            
            return: numpy array. Eq Points for each state.
        """
        
        eqpt = ct.find_eqpt(nonlin_myvehicle,x0,u0)
        
        return eqpt[0]

    def LinearizeSysMat(self, nonlin_myvehicle,u0 = [0,0,0] , x0 = [1,2,.5,.1,50,.1] ):
        """
            Return system matrix after linearizing at Eq point
            
            Input 1: NonLinearIO system type object. Open Loop System instance
            Input 2: List type. Intial input
            Input 3: List type. Initial position
            
            return 1: A matrix
            return 2: B matrix
            return 3: C matrix
            return 4: D matrix
            
            
        """
        
        xeq = self.FindEqPoint(nonlin_myvehicle,u0,x0)
        
        lin_myvehicle = ct.linearize(nonlin_myvehicle,xeq,0)
        
        A=lin_myvehicle.A
        B=lin_myvehicle.B
        C=lin_myvehicle.C
        D=lin_myvehicle.D
        
        return A,B,C,D
    
    

    def PolePlaceKMatrix(self, nonlin_myvehicle, poles = [-2,-10,-8, -6, -9, -8.5],u0 = [0,0,0] , x0 = [1,2,.5,.1,50,.1] ):
        """
        
            returns K matrix 
            
            Input 1: NonLinearIO system type object. Open Loop System instance
            Input 2: List type. Pole positions
            Input 3: List type. Intial input
            Input 4: List type. Initial position
            
            return 1: numpy array type. K matrix
            return 2: numpy array type. Knew matrix
            return 3: numpy array type. Kpos matrix
        
        
        """
        
        A,B,C,D = self.LinearizeSysMat(nonlin_myvehicle , u0 , x0)
        
        K = ct.place(A,B, poles)
        
        K3=K[:,2]
        K4=K[:,3]
        K6=K[:,5]

        K1=K[:,0]
        K2=K[:,1]
        K5=K[:,4]

        Knew = np.array([K3,K4,K6])
        Knew = np.transpose(Knew)

        Kpos = np.array([K1,K2,K5])
        Kpos = np.transpose(Kpos)
        
        return K, Knew, Kpos

    def StateFeedbackPosControl(self, nonlin_myvehicle, poles = [-2,-10,-8, -6, -9, -8.5], u0 = [0,0,0] , x0 = [1,2,.5,.1,50,.1]):
        
        """
            Close loop system with state feedback controller

            Input 1: NonLinearIO system type object. Open Loop System instance
            Input 2: List type. Pole positions
            Input 3: List type. Intial input
            Input 4: List type. Initial position

            return: feedback/IO system object. Close loop system with state feedback controller.
        """
        K, Knew, Kpos = self.PolePlaceKMatrix(nonlin_myvehicle, poles, u0, x0  )

        feed_nonlin_vehicle = ct.feedback(nonlin_myvehicle,K,sign=-1)

        return feed_nonlin_vehicle

    def StateFeedbackPosInputResponse(self, Timespan, Input = [1,1,1],  poles = [-2,-10,-8, -6, -9, -8.5], u0 = [0,0,0] , x0 = [1,2,.5,.1,50,.1], Xir = [1,0,0,0,0,0]):
        
        """
        Close loop system response

        
        Input 1: numpy array. Time vector
        Input 2: list type. unitary input
        Input 3: List type. Pole positions
        Input 4: List type. Intial input
        Input 5: List type. Initial position for Equilibrium point
        Input 6: List type. Initial position for time response

        return 1: 1D array type. response time vector
        return 2: (n x T)D array type where n = number of states. T = time period
        """


        # Input array
        inp = np.ones(Timespan.shape)
        R = np.array([inp*Input[0],inp*Input[1],inp*Input[2]])

        nonlin_myvehicle = self.OpenLoopSys()

        K, Knew, Kpos = self.PolePlaceKMatrix(nonlin_myvehicle, poles, u0, x0  )

        feed_nonlin_vehicle = self.StateFeedbackPosControl(nonlin_myvehicle, poles , u0 , x0)

        t, y = ct.input_output_response(feed_nonlin_vehicle, Timespan , Kpos@R, Xir)

        return t, y


# --------------------------------------CONTROLLER

    def GainMatK(self):

        # find equilibrium points
        u0 =np.array([0,0,0]) # no control 
        x0 = np.array([1,2,50,.5,.1,.1])
        eqpt = ct.find_eqpt(self.RawSystem(),x0,u0)
        xeq=eqpt[0]

        # Linearize the model at eq. points
        lin_myvehicle = ct.linearize(self.RawSystem(),xeq,0)
        A=lin_myvehicle.A
        B=lin_myvehicle.B
        C=lin_myvehicle.C 
        D=lin_myvehicle.D
    
    
        # Gain Matrix
    
        # Gain matrix K
        eigval = [-2,-10,-8, -6, -9, -8.5] # desired eigen values
        K = ct.place(A,B,eigval)

        Kp = np.array([K[:,0],K[:,1],K[:,2]])
        Kp = np.transpose(Kp)

        Kv = np.array([K[:,3],K[:,4],K[:,5]])
        Kv = np.transpose(Kv)
    
        return Kp,Kv,K

    # position controller
    def PosControlSystem(self):
       
        Kp,Kv,K = self.GainMatK()

        sysK  = ct.append(K)
        sysKp = ct.append(Kp)
        sysKv = ct.append(Kv)

        sysK  = ct.LinearIOSystem(sysK,inputs=['x1','x2','x3','x4','x5','x6'],outputs=['k1','k2','k3'],states=0)  # these systems has not dynamics. Therefore 0 states
        sysKp = ct.LinearIOSystem(sysKp,inputs=['rkp1','rkp2','rkp3'],outputs=['rpk1','rpk2','rpk3'],states=0)
        sysKv = ct.LinearIOSystem(sysKv,inputs=['rkv1','rkv2','rkv3'],outputs=['rvk1','rvk2','rvk3'],states=0)


        # Connect plant and state feedback
        feedsys = ct.feedback(self.RawSystem(),sysK,sign=-1)

        # Entire system for position control
        syspos = ct.series(sysKp,feedsys)

        # Entire system for velocity control
        sysvel = ct.series(sysKv,feedsys)
        return syspos

    # velocity constroller
    def VelControlSystem(self):

        Kp,Kv,K = self.GainMatK()

        sysK  = ct.append(K)
        sysKp = ct.append(Kp)
        sysKv = ct.append(Kv)

        sysK  = ct.LinearIOSystem(sysK,inputs=['x1','x2','x3','x4','x5','x6'],outputs=['k1','k2','k3'],states=0)  # these systems has not dynamics. Therefore 0 states
        sysKp = ct.LinearIOSystem(sysKp,inputs=['rkp1','rkp2','rkp3'],outputs=['rpk1','rpk2','rpk3'],states=0)
        sysKv = ct.LinearIOSystem(sysKv,inputs=['rkv1','rkv2','rkv3'],outputs=['rvk1','rvk2','rvk3'],states=0)


        # Connect plant and state feedback
        feedsys = ct.feedback(self.RawSystem(),sysK,sign=-1)

        # Entire system for position control
        syspos = ct.series(sysKp,feedsys)

        # Entire system for velocity control
        sysvel = ct.series(sysKv,feedsys)
        return sysvel



