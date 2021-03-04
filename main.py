# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# See PyCharm help at https://www.jetbrains.com/help/pycharm/


def start():
    testBowtie = createTestBowtie()
    trainingSetEventTree = createTrainingSetEventTree(testBowtie)
    trainingSetFaultTree = createTrainingSetFaultTree(testBowtie)
    topEvent = getTopEvent(testBowtie)
    learningParameters = getLearningParameters()

    undirectedEventTree = createUndirectedTree(trainingSetEventTree)
    undirectedFaultTree = createUndirectedTree(trainingSetFaultTree)
    directedEventTree = createDirectedTree(undirectedEventTree, topEvent)
    directedFaultTree = createDirectedTree(undirectedFaultTree, topEvent)
    bowtie = createBowTie(directedEventTree, directedFaultTree)
    

def createTestBowtie():
    print("createTestBowtie")
    return 1

def createTrainingSetFaultTree(testBowtie):
    print("createTrainingSetFaultTree")
    return 1

def createTrainingSetEventTree(testBowtie):
    print("createTrainingSetEventTree")
    return 1

def getTopEvent(testBowtie):
    print("getTopEvent")
    return 1

def getLearningParameters():
    print("getLearningParameters")
    return 1

def createUndirectedTree(trainingSet):
    print("createUndirectedTree")
    return 1

def createDirectedTree(undirectedTree,topEvent):
    print("createDirectedTree")
    return 1

def createQuantiviedTree(undirectedTree,topEvent):
    print("createQuantiviedTree")
    return 1


def createBowTie(directedEventTree, directedFaultTree):
    print("createBowTie")
    return 1



if __name__ == '__main__':
    start()


