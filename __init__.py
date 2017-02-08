# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

########################################################################
## Class Definitions ###################################################
########################################################################
class spectrum:
    def __init__(self,x = None,y = None,model = None,fitParams = None,paramInitValues = None,bestFitValues = None):
          
        # Fields
        self.x = x if x is not None else []
        self.y = y if y is not None else []
        self.label = []
        self.xlabel = ('Wavelength (nm)')
        self.ylabel = ('Intensity (A.U.)')
        self.model = model if model is not None else []
        self.fitParams = fitParams if fitParams is not None else []
        self.paramInitValues = paramInitValues if paramInitValues is not None else []
        self.bestFitValues = bestFitValues if bestFitValues is not None else []
      
    def __repr__(self):
        if self.x is not None:
            repstring = '<Spectrum|x ∈ [{}...{}]>'.format(self.x[0],self.x[-1])
        return repstring

    def plot(self,show='off'):
        plt.figure()
        plt.plot(self.x,self.y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if show in [1,'show','Show','On,''on','y','yes','Y','Yes']:
            plt.show()
        else:
            return plt.gca()
    
    # This function permanently trims the data
    def trim(self,lowerlim = None,upperlim = None):
        lidx = (np.abs(self.x-lowerlim)).argmin()
        uidx = (np.abs(self.x-upperlim)).argmin()+1 # This algorithm excludes the upper bound, for some odd reason (similar to a floor rounding)
        self.x = self.x[lidx:uidx]
        self.y = self.y[lidx:uidx]
        
    # This function returns a another spectrum that is sliced from the data
    def slice(self,lowerlim = None,upperlim = None):
        lidx = (np.abs(self.x-lowerlim)).argmin()
        uidx = (np.abs(self.x-upperlim)).argmin()+1 # This algorithm excludes the upper bound, for some odd reason (similar to a floor rounding)
        copy = self
        copy.x = copy.x[lidx:uidx]
        copy.y = copy.y[lidx:uidx]
        return copy
    
    def mean(self,lowerlim = None,upperlim = None):
        return np.mean(self.slice(lowerlim,upperlim).y)
        
class spectrum2D:
    def __init__(self,x = None,y = None,s = None):

        # Fields
        self.x = x if x is not None else []
        self.y = y if y is not None else []
        self.s = s if s is not None else [] # s should be the shape of (y,x)
        self.xlabel = ('Wavelength (nm)')
        self.ylabel = ('Time (ps)')
        self.zlabel = (r'$-\Delta T$')

    def __repr__(self):
        if self.x is not None:
            repstring = '<Spectrum2D|x ∈ [{}...{}]|y ∈ [{}...{}]>'.format(self.x[0],self.x[-1],self.y[0],self.y[-1])
        return repstring
    
    def slice(self,axis,startingVal,endingVal = None,valStepSize = None):
        # First check the axis
        if axis in ['x','X']:
            indices = findIndicesOfSlicedArray(self.x,startingVal,endingVal = None,valStepSize = None)
            for ind in indices
        
            
        if axis in ['y','Y']:
            

    def plot(self):
        plt.figure()
        plt.plot(self.x,self.y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()

########################################################################
## Function Definitions ################################################
########################################################################
def findIndexOfClosestValue(array,value):
    # This function takes an array and a desired value, and finds the 
    # index of the element whose value is closest to the desired value
    return min(range(len(array)), key=lambda i: abs(array[i]-value))
    
def findIndicesOfSlicedArray(array,startingVal,endingVal = None,valStepSize = None):
    # Convert the starting value to the index of the element along the axis
    startingInd = findIndexOfClosestValue(array,startingVal)
    
    # Did the user give an ending value?
    if endingVal is not None:
        # Convert the ending value to the index of the element along the axis
        endingInd = findIndexOfClosestValue(array,endingVal)
        
        # Did the user give a step size?
        if valStepSize is not None:
            # Convert the desired value step size into the appropriate index step size
            indStepSize = findIndexOfClosestValue(array,startingVal+valStepSize)-startingInd
            # arange does not include the last index
            indices = np.arange(startingInd,endingInd+indStepSize,indStepSize)
        else:
            indices = np.arange(startingInd,endingInd+1,1)
    else:
        indices = startingInd
    return indices

def load(filename,exptype):
    
    # for Cary UVVIS Absorption Spectrum data files
    if exptype in ['Cary', 'cary', 'CaryUVVis', 'CaryAbs', 'CaryAbsorption', 'abs', 'Abs']:
        # Create list for data to parse through after file is closed
        datalist = []
        # Open up file
        with open(filename) as f:
            # Go through it line by line
            for linenumber,line in enumerate(f):
                # Pick out the labels
                if linenumber == 0:
                    labels = line.split(',')[:-1][::2]
                    numberOfSpectra = len(labels)
                # Truncate the loading of data to hte split between the end of the data and the footer
                if line == '\r\n':
                    numberOfPoints = linenumber - 2
                    break
                # Create a list of all the data from the file
                if linenumber > 1:
                    # Add each line of data to the list
                    datalist.extend(line.split('\r\n'))
        # Remove the empty list values created by the newline characters
        datalist = datalist[::2]
        wavelength = np.zeros((numberOfSpectra,numberOfPoints))
        transmission = np.zeros((numberOfSpectra,numberOfPoints))
        # Arrange the data into an array
        for linenumber,line in enumerate(datalist):
            wavelength[:,linenumber] = [float(j) for j in line[:-1].split(',')[::2]]
            transmission[:,linenumber] = [float(j) for j in line[:-1].split(',')[1::2]]
        outputspectra = []
        for i in range(numberOfSpectra):
            tempspectrum = pss.spectrum(wavelength[i,:],transmission[i,:])
            tempspectrum.label = labels[i]
            outputspectra.append(tempspectrum)
            
    ### NEED TO ADD A DATA TYPE HERE for transmission type    
        
        return outputspectrum

########################################################################
## Tests  ##############################################################
########################################################################

if __name__=="__main__":
    # A = spectrum(np.array([1,2,3,4,5,6,7]),np.array([6,2,6,2,4,3,9]))
    # B = spectrum