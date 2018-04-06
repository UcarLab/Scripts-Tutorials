
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd

#Original Author: Asa Thibodeau

#TODO 1)code formatting, 2) better parameter names 3) Documentation/Error Messages

#Simple function to add 1 from the start position.
def convertToPositionFormatFromBED(data, startidx=1):
    cdata = np.copy(data)
    cdata[:,startidx] = cdata[:,startidx]+1
    return cdata

#Simple function to subtract 1 from the start position.
def convertToBEDFormatFromPosition(data, startidx=1):
    cdata = np.copy(data)
    cdata[:,startidx] = cdata[:,startidx]-1
    return cdata


###################################################################
#The functions below assume position format. (1-base fully-closed)#
###################################################################


#Returns a dictionary by chromosome where each element in the dictionary
#contains an array containing start positions sorted in ascending order
#and the position of the original data element containing the start position.
#Time complexity: O(nlogn), n=#peaks
def getChrStartSorted(data, chridx=0, startidx=1):
    allchr = np.unique(data[:,chridx])
    rv = dict()
    for i in range(0, len(allchr)):
        idx = np.where(data[:,chridx] == allchr[i])[0]
        chrdata = data[idx,:]
        sidx = np.argsort(chrdata[:,startidx],kind="mergesort")
        rv[allchr[i]] = np.concatenate((np.transpose(chrdata[sidx,startidx][np.newaxis]), np.transpose(idx[sidx][np.newaxis])), axis=1)
    return rv


#Returns the index of all regions that overlap the given chromosome, start, and end position.
#Time Complexity: O(logn), n = # of elements in the region list
def getOverlappingRegions(chrom, start, end, chrstartsorted, data, eidx=2):
    try:
        startsorted = chrstartsorted[chrom]
    except:
        startsorted = [] 
    s = 0
    e = len(startsorted)
    while (e-s) > 1:
        mi = s+((e-s)/2)
        mstart = startsorted[mi,0]
        if mstart < start:
            s = mi
        elif mstart > start:
            e = mi
        else:
            s = mi
            e = mi
    #scan until starts are greater than end
    rv = []
    idx = s
    while idx < len(startsorted) and end > startsorted[idx,0]:
        didx = startsorted[idx,1]
        cstart = startsorted[idx,0]
        cend = data[didx,eidx]
        if start <= cend and end >= cstart: #position format comparison
            rv.append(didx)
        idx = idx+1
    return rv


#Returns a boolean vector indicating whether or not the peak in the list overlaps a consensus peak.
#Time Complexity: O(nlogn), n = # of elements in the region list
def getOverlapIndex(data, consensuspeaks, chridx=0, startidx=1, endidx=2, cchridx=0, cstartidx=1):
    sortedconsensus = getChrStartSorted(consensuspeaks, cchridx, cstartidx)
    rv = np.zeros(len(data), dtype=bool)
    for i in range(0, len(data)):
        curchr = data[i, chridx]
        curstart = data[i, startidx]
        curend = data[i, endidx]
        if(len(getOverlappingRegions(curchr, curstart, curend, sortedconsensus, consensuspeaks, 2)) > 0):
            rv[i] = True
    return rv

#Count how many peaks & in which dataset the current peak list overlaps.
#Time Complexity: O(m*nlogn), n = # of elements in the region list, m = # of datasets
def getOverlapCount(countdataset, datasets, chridx=0, startidx=1, endidx=2):
    overlapvector = np.zeros(len(countdataset))
    overlapmatrix = np.zeros((len(countdataset), len(datasets)))
    for i in range(0, len(datasets)):
        curv = getOverlapIndex(countdataset, datasets[i], chridx, startidx, endidx, chridx, startidx).astype(int)
        overlapmatrix[:,i] = curv
        overlapvector = overlapvector+curv 
    return overlapvector, overlapmatrix


#Returns a list of consensus peaks using cascading overlap.
#Cascading overlap requires that all samples overlap at least one
#other sample within the region.
#Time complexity: O(n*m^2*logm + m*nlogn) m=#of datasets, n=#peaks
def getCascadingConsensusPeaks(data, chridx=0, startidx=1, endidx=2):
    sorteddata = dict()
    for i in range(0, len(data)):
        sorteddata[i] = getChrStartSorted(data[i], chridx, startidx)

    chromosomes = sorteddata[0].keys()
    for i in range(1, len(data)):
        chromosomes = np.union1d(chromosomes, sorteddata[i].keys())

    allchrcascadingpeaks = []
    for curchr in chromosomes:
        counters = np.zeros(len(data), dtype=np.int32)
        cascadingpeaks = []

        while True:
            curlist = []
            for i in range(0, len(data)):
                curdata = data[i]
                curindex = sorteddata[i][curchr][counters[i],1]
                curlist.append(curdata[curindex,:])
            curlist = np.array(curlist)
            slist = getChrStartSorted(curlist, chridx, startidx)

            order = slist[curchr][:,1]

            curindex = order[0]
            cascadestart = curlist[curindex,startidx]
            cascadeend = curlist[curindex,endidx]
            seenlist = [curindex]

            overlap = True
            for i in range(1, len(order)):
                curindex = order[i]
                curstart = curlist[curindex,startidx]
                curend = curlist[curindex,endidx]

                if curstart <= cascadeend: #position format comparison
                    cascadeend = max(curend, cascadeend)
                    seenlist.append(curindex)   
                else:
                    overlap = False
                    break

            #check previous cascading peak
            if len(cascadingpeaks) > 0 and cascadingpeaks[-1][2] >= cascadestart: #position format comparison
                cascadingpeaks[-1][2] = max(cascadeend, cascadingpeaks[-1][2])
            else:
                if overlap:
                    cascadingpeaks.append([curchr, cascadestart, cascadeend])

            #update counters
            for i in seenlist:
                counters[i] = counters[i]+1

            maxreached = False
            for i in range(0, len(counters)):
                if(counters[i] >= len(sorteddata[i][curchr])):
                    maxreached = True
                    break;

            if maxreached:
                #extend final cascading peak from remaining
                for i in range(0, len(counters)):
                    if(counters[i] < len(sorteddata[i][curchr])):
                        curstart = sorteddata[i][curchr][counters[i],0]
                        curidx = sorteddata[i][curchr][counters[i],1]
                        if(curstart <= cascadingpeaks[-1][2]): #position format comparison
                            cascadingpeaks[-1][2] = max(data[i][curidx,endidx], cascadingpeaks[-1][2])
                        else:
                            break
                break
        for cp in cascadingpeaks:
            allchrcascadingpeaks.append(cp)
            
    return np.array(allchrcascadingpeaks, dtype=object)


#Gets a strict set of consensus peaks requiring that every sample overlaps
#every other sample in the consensus peak region.
#Time Complexity: O(m^2*n + m*nlogn) m=#of datasets, n=#peaks 
def getStrictConsensusPeaks(data, chridx=0, startidx=1, endidx=2):
    sorteddata = dict()
    for i in range(0, len(data)):
        sorteddata[i] = getChrStartSorted(data[i], chridx, startidx)

    chromosomes = sorteddata[0].keys()
    for i in range(1, len(data)):
        chromosomes = np.union1d(chromosomes, sorteddata[i].keys())

    allchrstrictpeaks = []
    for curchr in chromosomes:
        counters = np.zeros(len(data), dtype=np.int32)
        strictpeaks = []

        while True:            
            curlist = []
            for i in range(0, len(data)):
                curdata = data[i]
                curindex = sorteddata[i][curchr][counters[i],1]
                curlist.append(curdata[curindex,:])
            curlist = np.array(curlist)
            
            #TODO get min and max positions
            minendidx = 0
            minend = curlist[0, endidx]
            maxstartidx = 0
            maxstart = curlist[0, startidx]
            for i in range(1, len(curlist)):
                curstart = curlist[i, startidx]
                if curstart > maxstart:
                    maxstart = curstart
                    maxstartidx = i
                curend = curlist[i, endidx]
                if curend < minend:
                    minend = curend
                    minendidx = i
            
            if maxstart < minend:
                #Add strict consensus peak
                strictpeaks.append([curchr, maxstart, minend])
                    
            #Remove the least end
            counters[minendidx] = counters[minendidx]+1
            

            maxreached = False
            for i in range(0, len(counters)):
                if(counters[i] >= len(sorteddata[i][curchr])):
                    maxreached = True
                    break;

            if maxreached:
                break
                
        for cp in strictpeaks:
            allchrstrictpeaks.append(cp)
            
    return np.array(allchrstrictpeaks, dtype=object)


