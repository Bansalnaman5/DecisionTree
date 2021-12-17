#code by NAMAN BANSAL
from sklearn import datasets
import pandas as pd
import math
from sklearn.model_selection import train_test_split
#class tree for building decision tree
class tree:
    def __init__(self,level,val,ent,gainratio,feat,leng):
        self.level=level
        self.val=val
        self.ent=ent
        self.gainratio=gainratio
        self.feat=feat
        self.leng=leng
        self.left=None
        self.right=None

#loading iris dataset and making a dataframe of same
isirsdata=datasets.load_iris()
xtrain,xtest,ytrain,ytest=train_test_split(isirsdata.data,isirsdata.target)
xdata=isirsdata.data
ydata=isirsdata.target
xcolumns=pd.DataFrame(xtrain)
#names of columns of dataset
global p
p=['sl','sw','pl','pw']
xcolumns.columns=p
#adding output column(res) in dataset
ycolumns=pd.DataFrame(ytrain)
ycolumns.columns=['res']
xcolumns['res']=ycolumns['res']
#function for splitting the dataframe according to best feature and condition on feature
#best feature is selected on bases of max info gain
def split(data,featur,comp):
    newframe=data[data[featur]<=comp]
    newframe1=data[data[featur]>comp]
    return newframe,newframe1
#function for calculating the entropy of dataset passed in function
def entropy(a):
    q=a['res'].value_counts()
    en=q.values
    l=len(a['res'])
    entrop=0
    for i in en:
        entrop+=-(i/l)*(math.log2((i/l)))
    return entrop

#recursive function for seleting the best feature and condition
#splitting the dataframe with max info gain
def select(xdataframe,lvl=0):
    #base condition 
    #if the entered node is pure then do not proceed further
    #make a tree node ,join and return
    if entropy(xdataframe)==0:
        t=tree(lvl,xdataframe['res'].value_counts(),1,None,None,None)
        return t
    #m for max value of info gain
    m=0
    parent_entropy=entropy(xdataframe)
    #total length of dataframe D
    D=len(xdataframe['res'])
    splitcondition=0
    bestfeature=""
    for j in p:
        #sorting the column of feature
        t=sorted(xdataframe[j])
        tot=len(xdataframe[j])
        l=[]
        #iterating on values of feature to find beast value
        for i in range(len(t)-1):
            #mid point of two features
            midpoint=(t[i]+t[i+1])/2
            #to avoid repetetion entering datapoint in array
            #operating only on dataset which is not encountered earlier
            if round(midpoint,2) not in l:
                k=round(midpoint,2)
                l.append(k)
                data1,data2=split(xdataframe,j,k)
                pre=parent_entropy-(((len(data1['res'])/tot)*entropy(data1))+((len(data2['res'])/tot)*entropy(data2)))
                #storing the value for feature and condintion after comparing info gains
                if round(pre,2)>m :
                    m=round(pre,2)
                    #if spl<k:
                    splitcondition=k
                    bestfeature=j
    #splitting dataframe on best feature and condition in two nodes
    node1,node2=split(xdataframe,bestfeature,splitcondition)
    #calculating splitinfo after the split
    g1=len(node1['res'])/D
    g2=len(node2['res'])/D
    g=(-g1*math.log2(g1))-(g2*math.log2(g2))
    G=m/g
    #print("gain ratio is:",G)
    #making node of tree
    root=tree(lvl,xdataframe['res'].value_counts(),parent_entropy,G,bestfeature,splitcondition)
    #recursive call on thesplitted dataframes and connecting them to make tree
    root.left=select(node1,lvl+1)
    root.right=select(node2,lvl+1)
    #returning the root
    return root
#function for printing the values of tree as specified in problem
def pri(ro):
    if ro==None:
        return
    print("Level:",ro.level)
    a=ro.val
    a1=a.index
    a2=a.values
    for i in range(len(a1)):
        print("Count of ",a1[i],"=",a2[i])
    print("Current Entropy is :",ro.ent)
    if ro.feat!=None:
        print("Splitting on feature :",ro.feat,"<=",ro.leng," with gain ratio :",ro.gainratio)
    else:
        print("Pure Node Reached")
    print()
    pri(ro.left)
    pri(ro.right)


#calling function to split and build tree
r=select(xcolumns)
#helper text for understanding codes 
print("Corresponding codes for classes are:")
print("0: Setosa")
print("1: Versicolor")
print("2: Verginica")
print()
#printing tree in the required format
pri(r)