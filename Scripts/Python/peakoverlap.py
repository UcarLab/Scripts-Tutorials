import numpy as np
import pandas as pd

#Author: Asa Thibodeau

def convertToPositionFormatFromBED(data, startidx=1):
    """Converts BED format positions to position format.

    Parameters
    ----------
    data : array-like, shape (n_peaks, n_features)
        A numpy array containing peak data in BED format.
        
    startidx : int
        The start index of the array. Default: 1
        
    Returns
    -------
    posdata : array-like, shape (n_peaks, n_features)
        Returns a copy of the array incrementing the start
        position by one.
    """
    posdata = np.copy(data)
    posdata[:,startidx] = posdata[:,startidx]+1
    return posdata

def convertToBEDFormatFromPosition(data, startidx=1):
    """Converts position format positions to BED format.

    Parameters
    ----------
    data : array-like, shape (n_peaks, n_features)
        A numpy array containing peak data in position format.
        
    startidx : int
        The start index of the array. Default: 1
        
    Returns
    -------
    beddata : array-like, shape (n_peaks, n_features)
        Returns a copy of the array decrementing the start
        position by one.
    """
    beddata = np.copy(data)
    beddata[:,startidx] = beddata[:,startidx]-1
    return cdata


###################################################################
#The functions below assume position format. (1-base fully-closed)#
###################################################################


def getChrStartSorted(data, chridx=0, startidx=1):
    """Returns a dictionary by chromosome where each element in
    the dictionary contains an array containing start positions
    sorted in ascending order and the position of the original
    data element containing the start position. 
    
    Time complexity: O(nlogn), n = # of peaks

    Parameters
    ----------
    data : array-like, shape (n_peaks, n_features)
        A numpy array containing peak data in position format.

    chridx : int
        The chromsome index of the array. Default: 0

    startidx : int
        The start index of the array. Default: 1
        
    Returns
    -------
    rv : dict
       A dictionary mapping each chromosome to an array:
       [0] = start position [1] = index in original data.
    """
    allchr = np.unique(data[:,chridx])
    rv = dict()
    for i in range(0, len(allchr)):
        idx = np.where(data[:,chridx] == allchr[i])[0]
        chrdata = data[idx,:]
        sidx = np.argsort(chrdata[:,startidx],kind="mergesort")
        rv[allchr[i]] = np.concatenate((np.transpose(chrdata[sidx,startidx][np.newaxis]), np.transpose(idx[sidx][np.newaxis])), axis=1)
    return rv


def getOverlappingRegions(chrom, start, end, chrstartsorted, data, eidx=2):
    """Returns the index of all regions that overlap the given
    chromosome, start, and end position.
    
    Time Complexity: O(logn), n = # of elements in the region list

    Parameters
    ----------
    chrom : str
        Chromsome of the position.

    start : int
        The start position.

    end : int
        The end position
        
    chrstartsorted : dict
        The chromosome start sorted dictionary of the regions
        to identify whether the given coordinates overlap.
    
    data : array-like, shape (n_peaks, n_features)
        The data corresponding to the sorted dictionary in
        positon format.
    
    eidx : int
        The end position indec within the data parameter. Default: 2
        
    Returns
    -------
    rv : tuple
       A tuple containing the index positions of data that
       overlap the given position.
    """
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
    return tuple(rv)


def getOverlapIndex(data, peakset, chridx=0, startidx=1, endidx=2, setchridx=0, setstartidx=1, setendidx=2):
    """Returns a boolean vector indicating whether or not the peak
    in the list overlaps a set of peaks.
    
    Time Complexity: O(nlogn), n = # of elements in the region list

    Parameters
    ----------
    data : array-like, shape (n_peaks, n_features)
        A numpy array containing peak data in position format.

        
    peakset : array-like, shape (n_peaks, n_features)
        A numpy array containing peak data in position format.

    
    chridx : int
        The chromsome index of the data parameter. Default: 0
        
    startidx : int
        The start index of the data parameter. Default: 1
        
    endidx : int
        The end index of the data parameter. Default: 2
        
    setchridx : int
        The chromosome index of the peakset parameter. Default: 0
    
    setstartidx : int
        The start index of the peakset parameter. Default: 1
        
    setendidx : int
        The end index of the peakset parameter. Default: 2
        
    Returns
    -------
    rv : arraylike, shape (n_peaks,)
       A boolean vector indicating whether the peaks in data overlap
       with the peaks in the peakset.
    """
    sortedconsensus = getChrStartSorted(peakset, setchridx, setstartidx)
    rv = np.zeros(len(data), dtype=bool)
    for i in range(0, len(data)):
        curchr = data[i, chridx]
        curstart = data[i, startidx]
        curend = data[i, endidx]
        if(len(getOverlappingRegions(curchr, curstart, curend, sortedconsensus, peakset, setendidx)) > 0):
            rv[i] = True
    return rv

def getOverlapCount(countdataset, datasets, chridx=0, startidx=1, endidx=2):
    """Counts how many peaks & in which dataset the current peak
    list overlaps over a set of peak lists.
    
    Time Complexity: O(m*nlogn)
    n = # of elements in the region list
    m = # of datasets

    Parameters
    ----------
    countdataset : array-like, shape (n_peaks, n_features)
        A numpy array containing peak data in position format.
        
    datasets : tuple
        A tuple containing multiple peak datasets in position format.

    chridx : int
        The chromsome index of the data parameter. Default: 0
        
    startidx : int
        The start index of the data parameter. Default: 1
        
    endidx : int
        The end index of the data parameter. Default: 2
        
    Returns
    -------
    overlapvector : arraylike, shape (n_peaks,)
       A vector indicating the number of datasets in datasets
       overlapping with the corresponding peak in countdataset.
       
    overlapmatrix : arraylike, shape (n_peaks, n_datasets)
       A boolean matrix indicating whether the peak countdataset
       overlaps with a peak in the corresponding dataset. Columns
       are ordered respective of the ordering in the dataset tuple.
    """
    overlapvector = np.zeros(len(countdataset))
    overlapmatrix = np.zeros((len(countdataset), len(datasets)))
    for i in range(0, len(datasets)):
        curv = getOverlapIndex(countdataset, datasets[i], chridx, startidx, endidx, chridx, startidx).astype(int)
        overlapmatrix[:,i] = curv
        overlapvector = overlapvector+curv 
    return overlapvector, overlapmatrix


def getCascadingConsensusPeaks(data, chridx=0, startidx=1, endidx=2):
    """Returns a list of consensus peaks using cascading overlap.
    Cascading overlap requires that all samples overlap at least
    one other sample within the region.
    
    Time complexity: O(n*m^2*logm + m*nlogn) m=#of datasets, n=#peaks

    Parameters
    ----------
    data : tuple
        A tuple of numpy arrays containing peak data in position format.
        
    chridx : int
        The chromsome index of the data parameter. Default: 0
        
    startidx : int
        The start index of the data parameter. Default: 1
        
    endidx : int
        The end index of the data parameter. Default: 2
        
    Returns
    -------
    rv : arraylike, shape (n_peaks,)
       A numpy array of genomic positions corresponding to the
       merged peak locations identified by cascading peaks across
       multiple positions.
    
    Notes
    -----
    Chromosome start and end positions must be the same over all peak
    datasets.
    """
    
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


def getStrictConsensusPeaks(data, chridx=0, startidx=1, endidx=2):
    """Returns a strict set of consensus peaks requiring that every
    sample overlaps every other sample in the consensus peak region.
    
    Time Complexity: O(m^2*n + m*nlogn) m=#of datasets, n=#peaks

    Parameters
    ----------
    data : tuple
        A tuple of numpy arrays containing peak data in position format.
        
    chridx : int
        The chromsome index of the data parameter. Default: 0
        
    startidx : int
        The start index of the data parameter. Default: 1
        
    endidx : int
        The end index of the data parameter. Default: 2
        
    Returns
    -------
    rv : arraylike, shape (n_peaks,)
       A numpy array of genomic positions where peaks across
       all datasets overlap.
    
    Notes
    -----
    Chromosome start and end positions must be the same over all peak
    datasets.
    """
        
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


def getAnnotationConfusionMatrix(data1, sorted1, data2, sorted2, 
                                 chr1idx=0, start1idx=1, end1idx=2, ann1idx=3,
                                 chr2idx=0, start2idx=1, end2idx=2, ann2idx=3):
    """Returns a confusion matrix of annotations between two sets of
        peaks at the base pair level.
    
    Time Complexity: O(n) n=#peaks

    Parameters
    ----------
    data1 : array-like
       Numpy array containing the peak data with annotations.
        
    sorted1 : dict
        Sort information for data1.
        
    data2 : array-like
       Numpy array containing the peak data with annotations.
        
    sorted2 : dict
        Sort information for data2.
        
    chr1idx : int
        The chromsome index of the data1 parameter. Default: 0    
        
    start1idx : int
        The start index of the data1 parameter. Default: 1
        
    end1idx : int
        The end index of the data1 parameter. Default: 2
        
    ann1idx : int
        The annotation index of the data1 parameter. Default: 3
        
    chr2idx : int
        The chromsome index of the data2 parameter. Default: 0    
        
    start2idx : int
        The start index of the data2 parameter. Default: 1
        
    end2idx : int
        The end index of the data2 parameter. Default: 2
        
    ann2idx : int
        The annotation index of the data2 parameter. Default: 3
        
    Returns
    -------
    m : arraylike, shape (n_unique_annotations1+1,n_unique_annotations2+1)
       A numpy array representing a confusion matrix between unique
       annotations among the two datasets to compare. An additional row
       and column is included for labels and unannotated regions.
    
    rowlabels : list
        The row labels of m (corresponding to data1).
    
    collabels : list
        The column labels of m (corresponding to data2).
    """
    
    ua1 = np.unique(data1[:, ann1idx])
    ua2 = np.unique(data2[:, ann2idx])
    chromosomes = np.union1d(sorted1.keys(),sorted1.keys())
    ua1l = len(ua1)
    ua2l = len(ua2)
    m = np.zeros((ua1l+1, ua2l+1), dtype=np.int64)
    rowlabels = list(ua1)
    rowlabels.append("NA")
    collabels = list(ua2)
    collabels.append("NA")
    
    for curchr in chromosomes:
        if curchr in sorted1 and curchr in sorted2:
            cursort1 = data1[list(sorted1[curchr][:,1]),:]
            cursort2 = data2[list(sorted2[curchr][:,1]),:]
            
            i1 = 0
            i2 = 0
            curpos = min(cursort1[i1,start1idx], cursort2[i1,start2idx])
            while(i1 < len(cursort1) and i2 < len(cursort2)):
                s1 = cursort1[i1,start1idx]
                e1 = cursort1[i1,end1idx]
                rowidx = np.where(ua1 == cursort1[i1,ann1idx])

                s2 = cursort2[i2,start2idx]
                e2 = cursort2[i2,end2idx]
                colidx = np.where(ua2 == cursort2[i2,ann2idx])
                if s1 <= e2 and s2 <= e1:


                    maxstart = max(s1,s2)
                    minend = min(e1,e2)

                    prefix = maxstart-curpos

                    if s1 < s2:
                        m[rowidx,ua2l] = m[rowidx,ua2l]+prefix
                    elif s2 < s1:
                        m[ua1l,colidx] = m[ua1l,colidx]+prefix

                    overlap = minend-maxstart+1
                    m[rowidx,colidx] = m[rowidx,colidx]+overlap

                    if(overlap < 0):
                        print str(s1)+":"+str(e1)+" "+str(s2)+":"+str(e2)

                    if e1 == minend:
                        i1 = i1+1
                    if e2 == minend:
                        i2 = i2+1
                    curpos = minend+1
                else:
                    if s2 > e1:
                        overlap = e1-s1+1
                        m[rowidx,ua2l] = m[rowidx,ua2l]+overlap
                        curpos = e1+1
                        i1 = i1+1
                    else:
                        overlap = e2-s2+1
                        m[ua1l,colidx] = m[ua1l,colidx]+overlap
                        curpos = e2+1
                        i2 = i2+1
                
            if e1 > minend:
                suffix = e1-curpos+1
                m[rowidx,ua2l] = m[rowidx,ua2l]+suffix
                i1 = i1+1
            elif e2 > minend:
                suffix = e2-curpos+1
                m[ua1l,colidx] = m[ua1l,colidx]+suffix
                i2 = i2+1
                
                
            while(i1 < len(cursort1)):
                pl = cursort1[i1,end1idx]-cursort1[i1,start1idx]+1 #assumes position format
                rowidx = np.where(ua1 == cursort1[i1,ann1idx])
                colidx = ua2l
                m[rowidx, colidx] = m[rowidx, colidx]+pl
                i1 = i1+1
                
            while(i2 < len(cursort2)):
                pl = cursort2[i2,end2idx]-cursort2[i2,start2idx]+1 #assumes position format
                rowidx = ua1l
                colidx = np.where(ua2 == cursort2[i2,ann2idx])
                m[rowidx, colidx] = m[rowidx, colidx]+pl
                i2 = i2+1
            
        else:
            if curchr in sorted1:
                cursort1 = data1[list(sorted1[curchr][:,1]),:]
                for j in range(0, len(cursort1)):
                    pl = cursort1[j,end1idx]-cursort1[j,start1idx]+1 #assumes position format
                    rowidx = np.where(ua1 == cursort1[j,ann1idx])
                    colidx = ua2l
                    m[rowidx, colidx] = m[rowidx, colidx]+pl
            else:
                cursort2 = data2[list(sorted2[curchr][:,1]),:]
                for j in range(0, len(cursort2)):
                    pl = cursort2[j,end2idx]-cursort2[j,start2idx]+1 #assumes position format
                    rowidx = ua1l
                    colidx = np.where(ua2 == cursort2[j,ann2idx])
                    m[rowidx, colidx] = m[rowidx, colidx]+pl
                    
    return m, rowlabels, collabels
