import numpy as np
import NavigationTrajectory as nt
import pandas as pd
import matplotlib.pyplot as plt

nt = nt.NavigationTrajectory()

class DataGenPrep:
    def __init__(self):
        print("DataGenPrep instance created")



    def ParseDict2Num(self,data):
        '''
        Input Parameters
        ----------
        data: dictionary type. Boundary conditions for range to generate data
        
        Returns
        
        Numpy Array of shape (8,2)
        
        '''
            
        i = len(list(data.keys()))
        
        array = np.zeros([i,2])
        
        for x in range(0,i):
            array[x] = data[list(data.keys())[x]]
            
        return array.astype(int)
    


    def Data_Array(self, BCrange,MaxIt):
        '''
        Input Arguments:
        BCrange:8x2 array type.
                contains boundary conditions in x direction.
                [initial boundary , Goal boundary ]
                [VI,VM1,VM2,VG,OI,OM1,OM2,OG1]

        MaxIt:  int type.
                number of observations. Batch size

        Returns:


        '''

        DataArray = np.zeros([MaxIt,16])

        i = 0
        j = 0
        k = 0

        while(i<MaxIt):
            while(j<16):

                if(j%2==0): # This is for x coordinates
                    DataArray[i,j] = np.random.randint(BCrange[k,0],BCrange[k,1])


                    k=k+1
                    j=j+1



                    #if(j==(BCrange.shape[0]-1)):
                    #k=0
                    #j=j+1
                    #else:
                    #k=k+1
                    #j=j+1



                else:# This is for y coordinates
                    DataArray[i,j] = np.random.randint(0,BCrange.max())

                    j=j+1

                    #if(j==BCrange.shape[0]-1):
                    #k=0
                    #j=j+1
                    #else:

                    #j=j+1


            j=0
            k=0
            i=i+1


        return DataArray.astype(int)
    

    def StaticRawInputTargetGen(self, myarray, MaxIt=10):
        '''
        Takes numpy array of Boundary range
        
        sacales the data and reuturns raw input for static model (VIX,VIY,VGX,VGY,OIX,OIY)
        
        return raw target (VIM1X,VIM1Y,VGM2X,VGM2Y)    
        
        parameter
            Bcrange array
        Return
            rawinp array, rawtarget array
        '''
        if(MaxIt>=10):
            dataset = self.Data_Array(myarray,MaxIt)
            scdata = self.Scale_Data(dataset)
            rawinp = np.array([scdata[:,0],scdata[:,1],scdata[:,6],scdata[:,7],scdata[:,8],scdata[:,9]])
            rawinp =rawinp.T
            rawtarget = np.array([scdata[:,2],scdata[:,3],scdata[:,4],scdata[:,5]])
            rawtarget = rawtarget.T

            return rawinp, rawtarget
        
        else:

            return print("Error: Maxit must be greater or equal to 10 for proper scaling of data!")


        

    


    def CSV_to_DataArray(self,file,array=True):
        '''    

        Parameters
        ----------
        file : CSV
            pass filename.csv as a string. file location must be current work directory
        array : boolean, default returns numpy array
            DESCRIPTION. 

        Returns
        -------
        TYPE
            csv data to numpy array
        TYPE
            csv data to pandas dataframe

        '''
        df = pd.read_csv(file)
        
        # Do random shuffle of rows/observations.
        df=df.sample(frac=1)
        df=df.reset_index(drop=True)
        
        Static_Input  = df.drop(columns=['VM1X1', 'VM1X2', 'VM2X1', 'VM2X2', 'OM1X1','OM1X2','OM2X1','OM2X2','OGX1','OGX2'])
        Static_Target = pd.DataFrame(df[['VM1X1', 'VM1X2', 'VM2X1', 'VM2X2']])
        
        if(array):
            myinput  = Static_Input.to_numpy() #np.asarray(Static_Input)
            mytarget = Static_Target.to_numpy()#np.asarray(Static_Target)
            
            return myinput,mytarget
        else:
            return Static_Input, Static_Target

    

    def Scale_Data(self, data):
        '''
        Scales data within 0 to 1 with min-max normalization

        x' = (x-min)/(max-min)

        Input Arguments:
        data: 
            numpy array of data
        
        Returns:
            numpy array of scaled data
        '''

        Scaled_Data = np.zeros([data.shape[0],data.shape[1]])

        i = 0
        maxcol = data.shape[1]

        while(i<maxcol):
            maxx = np.max(data[:,i])
            minn = np.min(data[:,i])
            Scaled_Data[:,i] = (data[:,i] - minn )/(maxx - minn)
            i=i+1
        
        return Scaled_Data
    

    def GaussianDist(self,x,mean,std):

        A = (1/(std*np.sqrt(2*np.pi)))
        
        B = (-.5)*((x-mean)/std)**2
        
        exp = np.exp(B)
        
        y = A*exp
        
        return y
    
    def GaussDistPlot(self,data,datacol, size=(15,15),m=4,n=4):


       

        fig, ax = plt.subplots(m, n, figsize=size)
        i=0
        j=0
        k=0
        while(i<4):
            while(j<4):
                Rand_Var = np.sort(data[:,k])
                mean = np.mean(Rand_Var )
                std  = np.std(Rand_Var )
                Gpdf = self.GaussianDist(Rand_Var,mean,std)
                ax[i,j].plot(Rand_Var, Gpdf,'r-', lw=5, alpha=0.6, label='norm pdf')
                ax[i,j].set_xlabel(datacol[k])
                k=k+1
                j=j+1
            
            i=i+1
            j=0
        
        return plt.show()
    

    def Filter_Good_Data(self, rawinp,rawtarget,parts=100,th=.1):
        '''
        

        Parameters
        ----------
        rawinp : numpy array
            DESCRIPTION.
        rawtarget : numpy array
            DESCRIPTION.
        parts : int, optional
            DESCRIPTION. The default is 100.
        th : float, optional
            DESCRIPTION. The default is .1. must be 0<th<1

        Returns
        -------
        training_data_input : numpy array
            DESCRIPTION.
        taining_data_target : numpy array
            DESCRIPTION.
        count : int
            DESCRIPTION.

        '''
        

        Vmove = nt.Generate_All_MoveC2(rawinp,rawtarget,parts)
        Opos= rawinp[:,4:6]
        count,index = nt.StaticCheckBad(Vmove,Opos,th)
        training_data_input = np.delete(rawinp,index.astype(np.int64),0)
        taining_data_target = np.delete(rawtarget,index.astype(np.int64),0)

        

        return training_data_input  , taining_data_target , count
    


    def Mux(self,Inp_Vec,Target_Vec,sel):
        if(sel == 0):
            return Inp_Vec,Target_Vec
        else:
            return np.array([]),np.array([])
    
    def SingleDataFilter(self,Inp_Vec,Target_Vec,Threshold, parts = 100):
        '''
        Input:

            Inp_Vec: Input array 
            Target_Vec: Target array
            Threshold: Cutoff value
            Parts: number of points for trajectory. Optional

        return: 

            Good input and target vector otherwise null array
        
        '''

        Vmove = nt.InputToTrajectory(Inp_Vec,Target_Vec,parts)

        Opos  = Inp_Vec[4:6]

        sel = nt.StaticCheckBad(Vmove,Opos,Threshold)


        GoodInput, GoodTarget = self.Mux(Inp_Vec,Target_Vec,sel) 

        return GoodInput, GoodTarget 
    

    def RandomGoodDataGenerator(self, myarray, MaxIteration, Threshold = .2 ,  parts = 100,  datapoint = 10 ):

    
        '''
        input:
            myarray: 8x2 array type.
                    contains boundary conditions in x direction.
                    [initial boundary , Goal boundary ]
                    [VI,VM1,VM2,VG,OI,OM1,OM2,OG1]
            
            MaxIteration: int type. number of desired good datapoints

            Threshold: between 0-1. boundary radius of obstacle

            parts: int type. Number of states/point to geenrate trajectory or move

            datapoints: int type, must be greater than or equal to 10 due to scaling. number of random data points need to be generated for filter 
        
        return:

            output1: array of good input array

            output2: array of good target array
        '''
        
        # List/container
        
        list_InpArray = []
        list_TarArray = []

        if(datapoint<10):
            
            return print("Error: datapoint can not be less than 10")

        else:

            while(len(list_InpArray)<=MaxIteration): # must be desired number of data
                
                # use StaticRawInputTargetGen  to generate raw_input and raw_target
                rawinp,rawtarget = self.StaticRawInputTargetGen(myarray,datapoint)

                # use RandomSingleIndexer to randomly select an input_vector and Target_vector
                indx = np.random.randint(0,datapoint) 
                
                # Use SingleDataFilter to generate good input target pair and store in a container        
                ginp,gtar = self.SingleDataFilter(rawinp[indx],rawtarget[indx],Threshold,parts)
                
                if(len(ginp)!=0): 
                    list_InpArray.append(ginp)
                    list_TarArray.append(gtar)
                
                
            
            return np.array(list_InpArray,dtype=float), np.array(list_TarArray,dtype=float)   
    



