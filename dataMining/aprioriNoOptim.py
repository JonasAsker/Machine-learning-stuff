import sys
import copy
import itertools as itr

def dataHandler(infile):
    """
    out - list of sets, the original indata 
    numTransactions - int, number of rows in the file
    """
    out = []
    numTransactions = 0
    with open(infile) as file:
        for l in file.readlines():
            lSplit = l.split("\t")
            lNums = list(map(lambda x: int(x), lSplit))
            out.append(set(lNums))
            numTransactions += 1
    return out, numTransactions


def getAllItems(dataSet):
    """
    dataSet - set of sets of ints, original indata
    itemsFreqs - dict with set of ints keys and int values, 
                sets of length one and their frequencies in the
                original input.
    """
    items = set()
    itemsFreqs = {}
    for i in dataSet:
        items = items.union(i)
    for i in items:
        for j in dataSet:
            if i in j:
                if frozenset({i}) in itemsFreqs.keys():
                    itemsFreqs[frozenset({i})] += 1
                else:
                    itemsFreqs[frozenset({i})] = 1
    return itemsFreqs


def getFreqs(items, dataSet):
    """
    items - list of sets
    dataset - list of sets
    out - dictionary with set keys and int values
    """
    out = dict()
    for i in items:
        for j in dataSet:
            if i.issubset(j):
                if (i not in out.keys()):
                    out[i] = 1
                else:
                    out[i] += 1
    return out


def pruneOnSupport(items, minSup, infrequentSets):
    """
    items - dict with set keys and int values
    minSup - int
    itemsCc - dict with set keys and int values, contains only 
            itemsets with frequency greater than the threshold
    infrequentSets - set of sets which have frequency smaller than the threshold
    """
    itemsCc = copy.copy(items)
    for i in items:
        if items[i] < minSup:
            del itemsCc[i]
            infrequentSets.add(i)
    return itemsCc, infrequentSets


def pruneOnSubset(candidate, infrequentSets):
    """
    candidate - a set of ints
    infrequentSets - a set of sets
    """
    for i in infrequentSets:
        if i.issubset(candidate):
            infrequentSets.add(candidate)
            return False
    return True


def candidateGeneration(prunedItems, infrequentItems):
    """
    prunedItems - dict with set of ints keys and int values
    out - set of sets of ints, generates candidates and prunes
          if the candidate has an infrequent subset.
    """
    items = list(prunedItems.keys())
    out = set()
    for i in prunedItems:
        for j in items[items.index(i):]:
            diff = len(i.difference(j))
            candidate = i.union(j)
            if diff == 1 and pruneOnSubset(candidate, infrequentItems):
                out.add(candidate)
    return out


def apriori(dataSet, items, minSup):
    """
    items - dict with set keys and int values
    prunedItems - dict with string keys and int values
    infrequentItems - set of strings
    candidates - set of sets of ints
    allFrequencies - dict with set of ints keys and int values which keeps track
                    of the support for all frequent sets found
    """
    infrequentItems = set()
    allFrequencies = dict()
    while True:
        prunedItems, infrequentItems = pruneOnSupport(items, minSup, infrequentItems)
        allFrequencies.update(prunedItems)
        candidates = candidateGeneration(prunedItems, infrequentItems)
        if not candidates:
            break
        items = getFreqs(candidates, dataSet)
    return allFrequencies


def associationRules(frequentPatterns, numTransactions):
    """
    frequentPatterns - dict with set of ints keys and int values, the final frequent
                    patterns and their supports
    """
    out = []
    for pattern in frequentPatterns:
        k = len(pattern)
        while k > 1:
            k -= 1
            subsets = itr.combinations(pattern, k)
            for ss in subsets:
                associativeSet = pattern.difference(ss)
                ss = frozenset(ss)
                support = round(100 * (frequentPatterns[pattern] / numTransactions), 2)
                conf = round(100 * (frequentPatterns[pattern] / frequentPatterns[ss]), 2)
                out.append((ss, associativeSet, support, conf))
    return out


def writeToFile(items, outfile):
    """
    items - dict with set of sets of ints keys and int values
    outfile - string, name of file to write to
    """
    with open(outfile, 'w') as file:
        for i in items:
            s = (str(i[0]).strip('frozenset()') + '\t'
                + str(i[1]).strip('frozenset()') + '\t'
                + str(i[2]) + '\t'
                + str(i[3]) + '\n')
            file.write(s)
            

def main():
    """
    dataset - list of sets of ints
    oneItemFreqs - dict with string keys and int values
    """
    minSup, infile, outfile = sys.argv[1:]
    dataSet, numTransactions = dataHandler(infile)
    minSup = (int(minSup) / 100) * numTransactions 
    oneItemFreqs = getAllItems(dataSet)
    frequentSets = apriori(dataSet, oneItemFreqs, minSup)
    associations = associationRules(frequentSets, numTransactions)
    writeToFile(associations, outfile)

main()