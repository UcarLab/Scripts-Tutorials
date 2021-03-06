import numpy as np
import pandas as pd
import peakoverlap as po

#Author: Asa Thibodeau

def combineSamples(datasets):
    """Combines multple samples into one.

    Parameters
    ----------
    datasets : tuple
        A tuple containing multiple numpy arrays with the same
        number of columns: 1: (n,d) 2: (m,d) etc.

    Returns
    -------
    rv : array-like, shape (n_samples, d)
        Returns an array concatenating all datasets.
    """
    return np.concatenate(datasets)

def readChrSizes(filepath):
    """Gets the chromosome sizes as a dictionary from a filepath.

    Parameters
    ----------
    filepath : str
        The filepath containing chromosome sizes in tab-delimited
        format: Chromosome\tChromosome Size.

    Returns
    -------
    rv : dict
        Returns a dictionary mapping chromosome to chromosome size.
    """
    values = pd.read_csv(filepath, sep="\t", header=None).values
    rv = dict()
    for i in range(0, len(values)):
        rv[values[i,0]] = values[i,1]
    return rv
    

def getRandomPartition(trainsamples, testsamples, randomstate=None, training=0.75, chrsizes=None):
    """Partitions data in train and test samples by randomly selecting peaks by
    a percentage for conensus peaks (identified using cascading approach) and
    combining them with the remaining peaks (those specific to training and testing)
    that are not within the consensus peak subset.

    Parameters
    ----------
    trainsamples : tuple
        A tuple containing multiple numpy arrays for the training set.

    testsamples : tuple
        A tuple containing multiple numpy arrays for the testing set.

    randomstate : int seed, or None(default)
        The seed to use for pseudo random number generation.

    training : float [0,1]
        The percentage used in training.  1-training will be used in
        the test set data. Values must be in the range [0,1].
        Default: 0.75
        
    chrsizes : Unused

    Returns
    -------
    trainrv : tuple,
        Returns a tuple containing the partitioned data indices
        for each training dataset.

    testrv : tuple
        Returns a tuple containing the partitioned data indices
        for each testing dataset.

    Notes
    -----
    Assumes that the first three columns of each data element are:
    Chromosome, Start, End
    """
    #use cascading peaks, identify them, select 75%, then add remaining from train/test samples
    allsamples = trainsamples+testsamples
    cascadingpeaks = po.getCascadingConsensusPeaks(allsamples, minoverlap=1)
    numcasc = len(cascadingpeaks)
    numtraining = int(np.floor(training*numcasc))
    
    np.random.seed(randomstate)
    trainindices = np.random.permutation(numcasc)[:numtraining]
    nulltestcascpeaks = cascadingpeaks[trainindices,:]
    
    bv = np.ones(numcasc, dtype=np.bool)
    bv[trainindices] = False
    nulltraincascpeaks = cascadingpeaks[bv,:]
    
    #MERGE peaks overlapping designated cascading peaks and peaks not overlapping them
    trainrv = []
    for i in range(0, len(trainsamples)):
        vector,_ = po.getOverlapCount(trainsamples[i], [nulltraincascpeaks])
        trainrv.append(np.where(vector == False)[0])
    
    testrv = []
    for i in range(0, len(testsamples)):
        vector,_ = po.getOverlapCount(testsamples[i], [nulltestcascpeaks])
        testrv.append(np.where(vector == False)[0])     
    return tuple(trainrv), tuple(testrv)


def getChromosomePortionPartition(trainsamples, testsamples, randomstate=None, training=0.75, chrsizes=None):
    """Partitions data in train and test samples by selecting peaks to be
    within the first fraction (specified by the training argument) to be in
    the training set and the remaining portion to be in the testing set.

    Parameters
    ----------
    trainsamples : tuple
        A tuple containing multiple numpy arrays for the training set.

    testsamples : tuple
        A tuple containing multiple numpy arrays for the testing set.

    chrsizes : dict
        A dictionary mapping chromsomes to their sizes.
        
    randomstate : unused

    training : float [0,1]
        The percentage of each chromosome (first half) used in training.
        1-training will be used in the test set data. Values must be in
        the range [0,1]. Default: 0.75
        
    chrsizes : unused

    Returns
    -------
    trainrv : tuple,
        Returns a tuple containing the partitioned data indices
        for each training dataset.

    testrv : tuple
        Returns a tuple containing the partitioned data indices
        for each testing dataset.

    Notes
    -----
    Assumes that the first three columns of each data element are:
    Chromosome, Start, End
    """
    limits = dict()
    for curchr in chrsizes.keys():
        limits[curchr] = training*chrsizes[curchr]

    trainrv = []
    for i in range(0, len(trainsamples)):
        vector = np.zeros(len(trainsamples[i]), dtype=np.bool)
        for j in range(len(trainsamples[i])):
            curchr = trainsamples[i][j,0]
            if trainsamples[i][j,2] < limits[curchr]:
                vector[j] = True
        trainrv.append(np.where(vector == True)[0])

    testrv = []      
    for i in range(0, len(testsamples)):
        vector = np.zeros(len(testsamples[i]), dtype=np.bool)
        for j in range(len(testsamples[i])):
            curchr = testsamples[i][j,0]
            if testsamples[i][j,2] >= limits[curchr]:
                vector[j] = True
        testrv.append(np.where(vector == True)[0]) 
    return tuple(trainrv), tuple(testrv)
    
    
def getOddEvenChromosomePartition(trainsamples, testsamples, randomstate=None, training=0.75, chrsizes=None):
    """Partitions data in train and test samples by selecting odd
    chromosomes for training samples and even chromosomes for testing
    samples.

    Parameters
    ----------
    trainsamples : tuple
        A tuple containing multiple numpy arrays for the training set.

    testsamples : tuple
        A tuple containing multiple numpy arrays for the testing set.
        
    randomstate : unused
    
    training : unused
    
    chrsizes : unused

    Returns
    -------
    trainrv : tuple,
        Returns a tuple containing the partitioned data indices
        for each training dataset.

    testrv : tuple
        Returns a tuple containing the partitioned data indices
        for each testing dataset.

    Notes
    -----
    Assumes that the first three columns of each data element are:
    Chromosome, Start, End
    """
    trainrv = []
    for i in range(0, len(trainsamples)):
        vector = np.zeros(len(trainsamples[i]), dtype=np.bool)
        for j in range(len(trainsamples[i])):
            curchr = trainsamples[i][j,0]
            try:
                chrnum = int(curchr[3:])
                if (chrnum % 2 == 1):
                    vector[j] = True
            except:
                pass
        trainrv.append(np.where(vector == True)[0])
    
    testrv = []       
    for i in range(0, len(testsamples)):
        vector = np.zeros(len(testsamples[i]), dtype=np.bool)
        for j in range(len(testsamples[i])):
            curchr = testsamples[i][j,0]
            try:
                chrnum = int(curchr[3:])
                if (chrnum % 2 == 0):
                    vector[j] = True
            except:
                pass
        testrv.append(np.where(vector == True)[0]) 
    return tuple(trainrv), tuple(testrv)


def getRandomByChromosomePartition(trainsamples, testsamples, randomstate=None, training=0.75, chrsizes=None):
    """Partitions data in train and test samples by randomly selecting peaks by
    a percentage for conensus peaks (identified using cascading approach) over each
    chromosome indepedently and combining them with the remaining peaks 
    (those specific to training and testing) that are not within the consensus peak
    subset.

    Parameters
    ----------
    trainsamples : tuple
        A tuple containing multiple numpy arrays for the training set.

    testsamples : tuple
        A tuple containing multiple numpy arrays for the testing set.

    randomstate : int seed, or None(default)
        The seed to use for pseudo random number generation.

    training : float [0,1]
        The percentage used in training.  1-training will be used in
        the test set data. Values must be in the range [0,1].
        Default: 0.75
        
    chrsizes : unused

    Returns
    -------
    trainrv : tuple,
        Returns a tuple containing the partitioned data indices
        for each training dataset.

    testrv : tuple
        Returns a tuple containing the partitioned data indices
        for each testing dataset.

    Notes
    -----
    Assumes that the first three columns of each data element are:
    Chromosome, Start, End
    
    If an integer random seed is provided, this number will be incremented
    for each call to getRandomPartition to ensure a different random number
    generation set for each chromosome.
    """
    sortedtrain = []
    sortedtest = []
    chrkeys = np.array([])
    for i in range(0, len(trainsamples)):
        sortedtrain.append(po.getChrStartSorted(trainsamples[i]))
        chrkeys = np.union1d(chrkeys, sortedtrain[i].keys())

    for i in range(0, len(testsamples)):
        sortedtest.append(po.getChrStartSorted(testsamples[i]))
        chrkeys = np.union1d(chrkeys, sortedtest[i].keys())

    trainbychrom = dict()
    testbychrom = dict()
    for curchr in chrkeys:
        trainsamplechr = []
        trainidx = []
        for i in range(0, len(trainsamples)):
            idx = sortedtrain[i][curchr][:,1]
            trainidx.append(idx)
            trainsamplechr.append(trainsamples[i][list(idx),:])

        testsamplechr = []    
        testidx = []        
        for i in range(0, len(testsamples)):
            idx = sortedtest[i][curchr][:,1]
            testidx.append(idx)
            testsamplechr.append(testsamples[i][list(idx),:])
            
        rstatei = None
        if randomstate != None:
            rstatei = randomstate+i
        curtrain, curtest =  getRandomPartition(trainsamplechr, testsamplechr, randomstate=rstatei, training=training)
        
        correctedtrain = []
        for i in range(0, len(curtrain)):
            correctedtrain.append(list(trainidx[i][curtrain[i]]))
            
        correctedtest = []
        for i in range(0, len(curtest)):
            correctedtest.append(list(testidx[i][curtest[i]]))
            
        trainbychrom[curchr] = correctedtrain
        testbychrom[curchr] = correctedtest

        
    trainrv = []
    for i in range(0, len(trainsamples)):
        trainrv.append(np.zeros((0, 0), dtype=np.int))
        for curchr in chrkeys:
            trainrv[i] = np.append(trainrv[i], trainbychrom[curchr][i])

    testrv = []
    for i in range(0, len(testsamples)):
        testrv.append(np.zeros((0, 0), dtype=np.int))
        for curchr in chrkeys:
            testrv[i] = np.append(testrv[i], testbychrom[curchr][i])
            
    return tuple(trainrv), tuple(testrv)


def getExclusivePartition(trainsamples, testsamples, randomstate=None, training=0.75, chrsizes=None):
    """Partitions data in train and test samples by selecting peaks
    in the training/testing set that are not in the corresponding
    testing/training set respectively.

    Parameters
    ----------
    trainsamples : tuple
        A tuple containing multiple numpy arrays for the training set.

    testsamples : tuple
        A tuple containing multiple numpy arrays for the testing set.
        
    randomstate : unused
    
    training : unused
    
    chrsizes : unused

    Returns
    -------
    trainrv : tuple,
        Returns a tuple containing the partitioned data indices
        for each training dataset.

    testrv : tuple
        Returns a tuple containing the partitioned data indices
        for each testing dataset.

    Notes
    -----
    Assumes that the first three columns of each data element are:
    Chromosome, Start, End
    """
    trainrv = []
    for i in range(0, len(trainsamples)):
        vector,_ = po.getOverlapCount(trainsamples[i], testsamples)
        trainrv.append(np.where(vector == 0)[0])
    
    testrv = []
    for i in range(0, len(testsamples)):
        vector,_ = po.getOverlapCount(testsamples[i], trainsamples)
        testrv.append(np.where(vector == 0)[0])       
    return tuple(trainrv), tuple(testrv)




######################################################################
#These partitions are meant to be baselines for inflated AUC curves  #
#that are due to testing on the same genomic regions as the training.#
######################################################################


def getIdentityPartition(trainsamples, testsamples, randomstate=None, training=0.75, chrsizes=None):
    """Doesn't partition the data at all and returns the input.

    Parameters
    ----------
    trainsamples : tuple
        A tuple containing multiple numpy arrays for the training set.

    testsamples : tuple
        A tuple containing multiple numpy arrays for the testing set.
        
    randomstate : unused
    
    training : unused
    
    chrsizes : unused

    Returns
    -------
    trainrv : tuple,
        Returns a tuple containing the partitioned data indices
        for each training dataset.

    testrv : tuple
        Returns a tuple containing the partitioned data indices
        for each testing dataset.

    Notes
    -----
    Assumes that the first three columns of each data element are:
    Chromosome, Start, End
    """
    trainrv = []
    for i in range(0, len(trainsamples)):
        trainrv.append(np.array(range(0,len(trainsamples[i]))))
    
    testrv = []
    for i in range(0, len(testsamples)):
        testrv.append(np.array(range(0,len(testsamples[i]))))
        
    return tuple(trainrv), tuple(testrv)


def getCascadingPartition(trainsamples, testsamples, randomstate=None, training=0.75, chrsizes=None, minoverlap=None):
    """Partitions data in train and test samples by selecting peaks
    overlapping cascading consensus peaks among the union of samples.

    Parameters
    ----------
    trainsamples : tuple
        A tuple containing multiple numpy arrays for the training set.

    testsamples : tuple
        A tuple containing multiple numpy arrays for the testing set.
        
    randomstate : unused
    
    training : unused
    
    chrsizes : unused

    Returns
    -------
    trainrv : tuple,
        Returns a tuple containing the partitioned data indices
        for each training dataset.

    testrv : tuple
        Returns a tuple containing the partitioned data indices
        for each testing dataset.

    Notes
    -----
    Assumes that the first three columns of each data element are:
    Chromosome, Start, End
    """
    allsamples = trainsamples+testsamples
    cascadingpeaks = po.getCascadingConsensusPeaks(allsamples, minoverlap=minoverlap)
    
    trainrv = []
    for i in range(0, len(trainsamples)):
        trainrv.append(np.where(po.getOverlapIndex(trainsamples[i], cascadingpeaks) == True)[0])
    
    testrv = []
    for i in range(0, len(testsamples)):
        testrv.append(np.where(po.getOverlapIndex(testsamples[i], cascadingpeaks) == True)[0])
        
    return tuple(trainrv), tuple(testrv)

def getStrictPartition(trainsamples, testsamples, randomstate=None, training=0.75, chrsizes=None):
    """Partitions data in train and test samples by selecting peaks
    overlapping strict consensus peaks among the union of samples.

    Parameters
    ----------
    trainsamples : tuple
        A tuple containing multiple numpy arrays for the training set.

    testsamples : tuple
        A tuple containing multiple numpy arrays for the testing set.
        
    randomstate : unused
    
    training : unused
    
    chrsizes : unused

    Returns
    -------
    trainrv : tuple,
        Returns a tuple containing the partitioned data indices
        for each training dataset.

    testrv : tuple
        Returns a tuple containing the partitioned data indices
        for each testing dataset.

    Notes
    -----
    Assumes that the first three columns of each data element are:
    Chromosome, Start, End
    """
    allsamples = trainsamples+testsamples
    strictpeaks = po.getStrictConsensusPeaks(allsamples)
    
    trainrv = []
    for i in range(0, len(trainsamples)):
        trainrv.append(np.where(po.getOverlapIndex(trainsamples[i], strictpeaks) == True)[0])
    
    testrv = []
    for i in range(0, len(testsamples)):
        testrv.append(np.where(po.getOverlapIndex(testsamples[i], strictpeaks) == True)[0])
        
    return tuple(trainrv), tuple(testrv)


