#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[5]:


print('--------\n CrI\u2083 : \n--------\n')
#!/usr/bin/python3.8
from datetime import datetime
start_time = datetime.now()
##-----------
import pickle as pk
import sys
from termcolor import colored
##--
import numpy as np
import math as mt
import scipy.linalg as la
from scipy.linalg import eigh
import scipy.interpolate
from itertools import groupby
import itertools
##----
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = [6.0, 5.0]
plt.rcParams["figure.autolayout"] = True
from matplotlib import cm
#matplotlib.use('Agg') ##-- Solving the plot problem
##---
from multiprocessing import Pool
import multiprocessing as mp
import concurrent.futures
from multiprocessing import cpu_count
processes = cpu_count()
##-------------------
## Useful Functions 
##-------------------
##-- Norm of Vectors
def Norm(vector1):
    return np.linalg.norm(vector1)
##---- function to convert superscript & subscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal),''.join(super_s))
    return x.translate(res)
def get_sub(x):
    normal = "n"
    sub_s = "ₙ"
    res = x.maketrans(''.join(normal),''.join(sub_s))
    return x.translate(res)
##---
##-------------------
##-- Constants
##-------------------
Spin = float("%.1f" %(3/2)) #spin
##-- Single-ion Anisotropy
A = -0.02 #for the SCF
##-- the Lande g factor
g = 2.002
##-- Boltzman Constant : unite is eV/Kelvin
B_C = 0.000086
KB  = float("%.3f" %(B_C*1000)) ##-- in meV/Kelvin
##-- The bohr magneton : unite is eV/tesla
bohr_magneton =0.000057
mu_B =float("%.3f" %(bohr_magneton*1000)) ##-- in meV/tesla
##-- Magnetic feild : unite is Tesla
B = 0.0
##-- Number of Unit cell
N_U = 1
##-------------------------------
##       UNIT CELL              
##-------------------------------
##-- LATTICE_VECTORS
a0 = 7.0  #lattice Constant
a = np.array([ a0,0.000,0.0000])
b = np.array([-a0*0.5,a0*0.86,0.0000])
c = np.array([ 0.000,0.000,12.000])
##-- Atoms Position in Unit Cell
A_P_U_C=np.array(
        [[0.555, 0.777, 0.000] # Atom 1
        ,[0.888, 0.444, 0.000] # Atom 2
        ])
##-- Number of Magnetic Sites in one Unit cell
N_M_S = A_P_U_C.shape[0]
##-------------------------------
##       SUPERCELL             
##-------------------------------
##-- LATTICE_VECTORS
a_m = 3 #a_multiple_by
b_m = 3 #b_multiple_by
c_m = 1 #c_multiple_by
a1  = a*a_m
a2  = b*b_m
a3  = c*c_m
##-- Atoms Position in Supercell
A_P_S = np.zeros([N_M_S,a_m*b_m*c_m,3])
for i in range(A_P_S.shape[0]):
        # Initail Position in Supercell
        A_P_S[i,0] = np.array([
                     float("%.3f" %(A_P_U_C[i,0]/a_m)),
                     float("%.3f" %(A_P_U_C[i,1]/b_m)),
                     float("%.3f" %(A_P_U_C[i,2]/c_m))
                              ])
        for j in range(1,a_m*b_m):
            if j < a_m : # Finding the Position in a1 Direction
                A_P_S[i,j] = A_P_S[i,0]+ np.array([float("%.3f" %(j/a_m)),0.0,0.0]) 
            else : ##-- Finding the Position in a2 Direction
                A_P_S[i,j] = A_P_S[i,j-a_m]+ np.array([0.0,float("%.3f" %(1/b_m)),0.0])
##-------------------------------
##    Neighbors and distance
##-------------------------------
##-- Cartesian Position from Atoms of Supercell
C_P = np.zeros([N_M_S*a_m*b_m,3])
l=0
for j in range(a_m*b_m):
    C_P[l] = A_P_S[0,j,0]*a1+A_P_S[0,j,1]*a2+A_P_S[0,j,2]*a3  #
    C_P[l+1] =A_P_S[1,j,0]*a1+A_P_S[1,j,1]*a2+A_P_S[1,j,2]*a3
    l=l+2 #Counting both atoms
##-- Sites
S1 = int((C_P.shape[0]/2)-1)      #Site 1 from centeral unit cell
S2 = int((C_P.shape[0]/2)-1) + 1  #Site 2 from centeral unit cell
dist=[] #distance between neighbors
for i in range(C_P.shape[0]):
    if i ==  S1:
        continue
    dist.append(float("%.4f" %(Norm(C_P[i]-C_P[S1]))))
dist=sorted(dist)
def nearby_groups(arr, tol_digits=0): #Sorting the list of distance
  # split up sorted input array into groups if they're similar enough
  for (_, grp) in groupby(arr, lambda x: round(x, tol_digits)):
    # yield value from group that is closest to an integer
    yield sorted(grp, key=lambda x: abs(round(x) - x))[0]
array = dist
dist=list(nearby_groups(array))
d1=dist[0]
d2=dist[1]
d3=dist[2]
d4=dist[3]
epsilon = 0.1
##-- Counting the Neighbors
D=0
J1 = 0
J2 = 0
J3 = 0
R_J1 = []
R_J2 = []
R_J3 = []
for i in range(C_P.shape[0]):
    if i != S1:
        D = float("%.4f" %(Norm(C_P[i]-C_P[S1])))
        if abs(D-d1) < epsilon :
            J1=J1+1
            R_J1.append(C_P[i]-C_P[S1])
            #print('r1 = '+ str(C_P[i]-C_P[S1]))
        if abs(D-d2) < epsilon :
            J2=J2+1
            R_J2.append(C_P[i]-C_P[S1])
            #print('r2 = '+ str(C_P[i]-C_P[S1]))
        if abs(D-d3) < epsilon :
            J3=J3+1
            R_J3.append(C_P[i]-C_P[S1])
            #print('r3 = '+ str(C_P[i]-C_P[S1])+'\n-------------')
del R_J2[3:]
# print('-----------Neighbors vectors FM Configuration---------------\n')
# print('The First  Neighbor = ' + str(R_J1) + '\n'
#      +'The Second Neighbor = ' + str(R_J2) + '\n'
#      +'The Third  Neighbor = ' + str(R_J3))
# print('---------------------------------------------------------------------\n')
print('-----------------------------------------------------------\n'
      +'The Nearest Neighbors for each site in FM Configuration :\n'
      +'The 1{} Neighbor = '.format(get_super('st')) + str(J1) + '\n'
      +'The 2{} Neighbor = '.format(get_super('nd')) + str(J2) + '\n'
      +'The 3{} Neighbor = '.format(get_super('rd')) + str(J3) + '\n'
      +'--------------------------------------------\n')
##------------------------
## K space griding (Start)
##------------------------
print('\n'+'Calculating the 1BZ : '+'\n')
##-- Reciprocal Lattice Vectors Unit Cell
b1 = (2*np.pi*(np.cross(b,c)))/np.dot(a,np.cross(b,c))
b2 = (2*np.pi*(np.cross(c,a)))/np.dot(a,np.cross(b,c))
b3 = (2*np.pi*(np.cross(a,b)))/np.dot(a,np.cross(b,c))
print('----------- Unit Cell Vectors in Reciprocal Space ------------'+'\n')
print( "b\u2081 = {}".format(b1)+'\n'
      +"b\u2082 = {}".format(b2)+'\n'
      +"b\u2083 = {}".format(b3)+'\n'
      +'---------------------------------------------'+'\n')
##-----------
##-- The heigh path is Gamma(0,0,0)--->K(0,0.5,0)--->M(0.5,0,0)--->Gamma(0,0,0)
G  = (0.0*b1)      + (0.0*b2)      + (0.0*b3) ##--- Gamma
K1 = ((1./3)*b1)   + ((1./3)*b2)   + (0.0*b3)
K2 = ((-1./3)*b1)  + ((2./3)*b2)   + (0.0*b3)
K3 = (((-2./3)*b1) + ((1./3)*b2)   + (0.0*b3))
K4 = ((-1./3)*b1)  + ((-1./3)*b2)  + (0.0*b3) ##--- (-K1)
K5 = ((1./3)*b1)   + ((-2./3)*b2)  + (0.0*b3) ##--- (-K2)
K6 = ((2./3)*b1)   + ((-1./3)*b2)  + (0.0*b3) ##--- (-K3)
M  = (0.5*b1)      + (0.0*b2)      + (0.0*b3) ##--- M
##--    2D plot of 1BZ
sd_Kx = int(input("Enter your K\u2081 subdivisions : ")) #subdivisions Kx
sd_Ky = int(input("Enter your K\u2082 subdivisions : ")) #subdivisions Ky
sd_Kz = int(input("Enter your K\u2083 subdivisions : ")) #subdivisions Kz
print('\n---------------------------------------------')
##--------------- Ploting the 1st unite cell in reciprocal space
def Plot_1st_U_R_S(K_Space):
    plt.plot(K_Space[:,0],K_Space[:,1],'k.')
    ##-- Gamma Point
    plt.text(G[0],G[1]+.02,'\u0393',color='red' , fontsize=20)
    plt.plot(G[0],G[1],'ro')
    ##-- K1 Point
    plt.text(K1[0]-.08, K1[1]-.08,"$K1$",color='red', fontsize=20)
    plt.plot(K1[0],K1[1],'ro')
    ##-- K2 prime Point
    plt.text(K2[0]+.01, K2[1]-.06,"$K2$",color='red', fontsize=20)
    plt.plot(K2[0],K2[1],'ro')
    ##-- K3 Point
    plt.text(K3[0]+.03, K3[1],"$K3$",color='red', fontsize=20)
    plt.plot(K3[0],K3[1],'ro')
    ##-- K4 Point
    plt.text(K4[0]+.01, K4[1]+.01,"$K4$",color='red', fontsize=20)
    plt.plot(K4[0],K4[1],'ro')
    ##-- K5 Point
    plt.text(K5[0]-.08, K5[1]+.03,"$K5$",color='red', fontsize=20)
    plt.plot(K5[0],K5[1],'ro')
    ##-- K6 Point
    plt.text(K6[0]-.11, K6[1]-.03,"$K6$",color='red', fontsize=20)
    plt.plot(K6[0],K6[1],'ro')
    ##--
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")
    plt.title('K points in the 1BZ')
#     plt.savefig("1_Unit_cell_in_RS.pdf")
#     plt.close()
    plt.show()
    return
##--------------- Ploting the 1st unite cell in reciprocal space
def Plot_1BZ_3D(K_Space):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ##-----------------
    x = K_Space[:,0]
    y = K_Space[:,1]
    ##-----------------
    ax.scatter(x, y, c='m', marker='*')
    ax.view_init(elev=90., azim=-90)
    ax.set_xticks([round(min(x),1),0.0,round(max(x),1)])
    ax.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ax.set_zticks([])
    #-----------
    ax.set_xlabel("$K_{x}$")
    ax.set_ylabel("$K_{y}$")
    ##-- Gamma Point
    ax.text(G[0]+.02,G[1]+.02,G[2]+.02,'\u0393',color='black' , fontsize=10)
    ax.scatter(G[0], G[1], c='k', marker='o')
    ##-- K1 Point
    ax.text(K1[0]-.08, K1[1]-.08, K1[2],"$K_1$",color='black', fontsize=10)
    ax.scatter(K1[0], K1[1], c='k', marker='o')
    ##-- K2 prime Point
    ax.text(K2[0]+.01, K2[1]-.06, K2[2]+.01,"$K_2$",color='black', fontsize=10)
    ax.scatter(K2[0], K2[1], c='k', marker='o')
    ##-- K3 Point
    ax.text(K3[0]+.03, K3[1],K3[2],"$K_3$",color='black', fontsize=10)
    ax.scatter(K3[0], K3[1], c='k', marker='o')
    ##-- K4 Point
    ax.text(K4[0]+.01, K4[1]+.01,K4[2]+.01,"$K_4$",color='black', fontsize=10)
    ax.scatter(K4[0], K4[1], c='k', marker='o')
    ##-- K5 Point
    ax.text(K5[0]-.08, K5[1]+.03,K5[2]+.01,"$K_5$",color='black', fontsize=10)
    ax.scatter(K5[0], K5[1], c='k', marker='o')
    ##-- K6 Point
    ax.text(K6[0]-.11, K6[1]-.03,K6[2]-.01,"$K_6$",color='black', fontsize=10)
    ax.scatter(K6[0], K6[1], c='k', marker='o')
    ##-- M Point
    ax.text(M[0]+.01, M[1]+.01,M[2]+.01,"$M$",color='green', fontsize=10)
    ax.scatter(M[0], M[1], c='g', marker='o')
    ##--
    ax.plot([G[0],K1[0]], [G[1], K1[1]],[G[2], K1[2]])
    ax.plot([G[0],M[0]],  [G[1], M[1]], [G[2], M[2]])
    ##--
    ax.set_title('K points in the 1BZ')
#     plt.savefig("1st_BZ_3D.pdf")
#     plt.close()
    plt.show()
    return
def Plot_1BZ():
    x1 = []
    y1 = []
    for i in range(0,sd_Kx,1):
        for j in range(0,sd_Ky,1):
            x1.append(K_BZ1[(i*sd_Kx)+j,0])
            y1.append(K_BZ1[(i*sd_Kx)+j,1])
    ##-- Gamma Point
    plt.text(G[0]-.02,G[1]+.02,'\u0393',color='blue' , fontsize=20)
    plt.plot(G[0],G[1],'ro')
    ##-- K1 Point
    plt.text(K1[0]-.08, K1[1]-.08,"$K1$",color='blue', fontsize=20)
    plt.plot(K1[0],K1[1],'ro')
    ##-- K2 Point
    plt.text(K2[0]+.01, K2[1]-.06,"$K2$",color='blue', fontsize=20)
    plt.plot(K2[0],K2[1],'ro')
    ##-- K3 Point
    plt.text(K3[0]+.03, K3[1],"$K3$",color='blue', fontsize=20)
    plt.plot(K3[0],K3[1],'ro')
    ##-- K4 Point
    plt.text(K4[0]+.01, K4[1]+.01,"$K4$",color='blue', fontsize=20)
    plt.plot(K4[0],K4[1],'ro')
    ##-- K5 Point
    plt.text(K5[0]-.08, K5[1]+.03,"$K5$",color='blue', fontsize=20)
    plt.plot(K5[0],K5[1],'ro')
    ##-- K6 Point
    plt.text(K6[0]-.11, K6[1]-.03,"$K6$",color='blue', fontsize=20)
    plt.plot(K6[0],K6[1],'ro')
    ##-- K6 Point
    plt.text(M[0]+.01, M[1]+.01,"$M$",color='green', fontsize=20)
    plt.plot(M[0],M[1],'go')
    ##-- lable of axis
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")
    ##-- Arrow plots
    plt.arrow(K1[0],K1[1],(K2-K1)[0],(K2-K1)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K2[0],K2[1],(K3-K2)[0],(K3-K2)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K3[0],K3[1],(K4-K3)[0],(K4-K3)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K4[0],K4[1],(K5-K4)[0],(K5-K4)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K5[0],K5[1],(K6-K5)[0],(K6-K5)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K6[0],K6[1],(K1-K6)[0],(K1-K6)[1],head_width = 0.0001, width = 0.002,ec ='black')
    ##
    plt.arrow(G[0], G[1], K1[0], K1[1],head_width = 0.0001, width = 0.002,ec ='red')
    plt.arrow(G[0], G[1], M[0], M[1], head_width = 0.0001, width = 0.002,ec ='green')
    plt.grid()
    plt.title('1BZ and High Symmetry Path')
    ##-- savinf plot
#     plt.savefig("1BZ.pdf")
#     plt.close()
    plt.show()
    return
###-------------------------
print('\n---------------------\n Sampling the 1BZ : \n---------------------')
##-- b1 > 0 & b2 > 0
def K_mesh1(i,j):
    K_BZ = (i/sd_Kx)*(b1)+(j/sd_Ky)*(b2)
    return K_BZ
if __name__ == '__main__':
    # Define the dataset
    dataset = [(i,j) for i in np.arange(0,sd_Kx+1,1) for j in np.arange(0,sd_Ky+1,1)]
    cpus = mp.cpu_count()
    with Pool(processes=cpus) as pool:
        res1 = pool.starmap(K_mesh1, dataset)
K_BZ1 =np.array(res1)
##-- b1 < 0 & b2 > 0
def K_mesh2(i,j):
    K_BZ = (i/sd_Kx)*(-b1)+(j/sd_Ky)*(b2)
    return K_BZ
if __name__ == '__main__':
    # Define the dataset
    dataset = [(i,j) for i in np.arange(1,sd_Kx+1,1) for j in np.arange(0,sd_Ky+1,1)]
    cpus = mp.cpu_count()
    with Pool(processes=cpus) as pool:
        res2 = pool.starmap(K_mesh2, dataset)
K_BZ2 =np.array(res2)
##-- b1 > 0 & b2 < 0
def K_mesh3(i,j):
    K_BZ = (i/sd_Kx)*(b1)+(j/sd_Ky)*(-b2)
    return K_BZ
if __name__ == '__main__':
    # Define the dataset
    dataset = [(i,j) for i in np.arange(0,sd_Kx+1,1) for j in np.arange(1,sd_Ky+1,1)]
    cpus = mp.cpu_count()
    with Pool(processes=cpus) as pool:
        res3 = pool.starmap(K_mesh3, dataset)
K_BZ3 = np.array(res3)
##-- b1 < 0 & b2 < 0
def K_mesh4(i,j):
    K_BZ = (i/sd_Kx)*(-b1)+(j/sd_Ky)*(-b2)
    return K_BZ
if __name__ == '__main__':
    # Define the dataset
    dataset = [(i,j) for i in np.arange(1,sd_Kx+1,1) for j in np.arange(1,sd_Ky+1,1)]
    cpus = mp.cpu_count()
    with Pool(processes=cpus) as pool:
        res4 = pool.starmap(K_mesh4, dataset)
K_BZ4 =np.array(res4)
#-- finding the K points in the 1BZ
K_BZ = np.zeros([(K_BZ1.shape[0]+K_BZ2.shape[0]+K_BZ3.shape[0]+K_BZ4.shape[0]),3])
K_BZ[0:K_BZ1.shape[0]] = K_BZ1
K_BZ[K_BZ1.shape[0]:K_BZ1.shape[0]+K_BZ2.shape[0]] = K_BZ2
K_BZ[K_BZ1.shape[0]+K_BZ2.shape[0]:K_BZ1.shape[0]+K_BZ2.shape[0]+K_BZ3.shape[0]] = K_BZ3
K_BZ[K_BZ1.shape[0]+K_BZ2.shape[0]+K_BZ3.shape[0]:] = K_BZ4
K_BZ_intial = K_BZ
K_BZ_Beery_plot = K_BZ
#---------------
K_BZ_x_limit = []
for i in K_BZ:
    if (i[0] <= round(K6[0],3) and i[0] >= 0) :
        K_BZ_x_limit.append(i)
    if (i[0] >= round(K3[0],3) and i[0] < 0):
        K_BZ_x_limit.append(i)
K_BZ_y_limit = []
for i in K_BZ_x_limit:
    if (i[1] <= round(K2[1],3) and i[1] > 0 ) :
        K_BZ_y_limit.append(i)
    if (i[1] >= round(K5[1],3) and i[1] < 0):
        K_BZ_y_limit.append(i)
K_BZ = np.zeros([len(K_BZ_y_limit),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_y_limit[i]
# ###-------------------------------------------------
# ##-- Definig the Equation of lines for the path
# ###--------------------------------------------
#--K3_K2
K_BZ_K3_K2 = []
def K3_K2(vec): 
    if K3[0]==K2[0] :
        K3K2 = abs(vec[0]-K3[0])
    elif K3[1]==K2[1] :
        K3K2 = abs(vec[1]-K3[1])
    else :
        K3K2 = vec[1]-(((K3[1]-K2[1])/(K3[0]-K2[0]))*(vec[0]-K2[0])+K2[1])
    return K3K2
for i in K_BZ_y_limit:
    if float("%.4f" %(K3_K2(i))) < 0:
        K_BZ_K3_K2.append(i)
    if float("%.4f" %(K3_K2(i))) == 0:
        K_BZ_K3_K2.append(i)
K_BZ = np.zeros([len(K_BZ_K3_K2),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_K3_K2[i]
#--K4_K3
K_BZ_K4_K3 = []
def K4_K3(vec): 
    if K4[0]==K3[0] :
        K4K3 = abs(vec[0]-K4[0])
    elif K4[1]==K3[1] :
        K4K3 = abs(vec[1]-K4[1])
    else :
        K4K3 = vec[1]-(((K4[1]-K3[1])/(K4[0]-K3[0]))*(vec[0]-K3[0])+K3[1])
    return K4K3
##--
for i in K_BZ_K3_K2:
    if float("%.4f" %(K4_K3(i))) > 0:
        K_BZ_K4_K3.append(i)
    if float("%.4f" %(K4_K3(i))) == 0:
        K_BZ_K4_K3.append(i)
K_BZ = np.zeros([len(K_BZ_K4_K3),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_K4_K3[i]
#--K6_K5
K_BZ_K6_K5 = []
def K6_K5(vec): 
    if K6[0]==K5[0] :
        K6K5 = abs(vec[0]-K6[0])
    elif K6[1]==K5[1] :
        K6K5 = abs(vec[1]-K6[1])
    else :
        K6K5 = vec[1]-(((K6[1]-K5[1])/(K6[0]-K5[0]))*(vec[0]-K5[0])+K5[1])
    return K6K5
##--
for i in K_BZ_K4_K3:
    if float("%.4f" %(K6_K5(i))) > 0:
        K_BZ_K6_K5.append(i)
    if float("%.4f" %(K6_K5(i))) == 0:
        K_BZ_K6_K5.append(i)
K_BZ = np.zeros([len(K_BZ_K6_K5),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_K6_K5[i]
#--K6_K5
K_BZ_K1_K6 = []
def K1_K6(vec): 
    if K1[0]==K6[0] :
        K1K6 = abs(vec[0]-K1[0])
    elif K1[1]==K6[1] :
        K1K6 = abs(vec[1]-K1[1])
    else :
        K1K6 = vec[1]-(((K1[1]-K6[1])/(K1[0]-K6[0]))*(vec[0]-K6[0])+K6[1])
    return K1K6
##--
for i in K_BZ_K6_K5:
    if float("%.4f" %(K1_K6(i))) < 0:
        K_BZ_K1_K6.append(i)
    if float("%.4f" %(K1_K6(i))) == 0:
        K_BZ_K1_K6.append(i)
K_BZ = np.zeros([len(K_BZ_K1_K6),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_K1_K6[i]
##--------
Plot_1st_U_R_S(K_BZ_intial)
Plot_1st_U_R_S(K_BZ)
Plot_1BZ_3D(K_BZ)
##------------------
## End of BZ
##------------------
Plot_1BZ()      ##-- ploting the 1BZ
print('1BZ is calculated.\n')
##------------------------
## K space griding (End)
##------------------------
##---------------------
##-- Exchange Interaction parameter for the first, second,third neighbors [J1,J2,J3] 
##---------------------
J=[-1.48, -0.08, 0.1]
##-- The sum of all J’s on one specific spin site
J_tilda = (J1*J[0])+(J2*J[1])+(J3*J[2])
E_g = N_M_S*N_U*((-J_tilda*(Spin**2))-(Spin*g*mu_B*B)-(A*(Spin**2)))##-- Ground state energy
##-- DMI coeficient
DMI = [0.0,0.0,0.0]
##-- phase
if J[1] == 0 and DMI[1]== 0 :
    phase = 0.0
elif J[1] == 0 :
    phase =np.pi/2 # if J2 = 0.0
elif J[1] != 0 :
    phase = np.arctan(DMI[1]/(J[1]))
print('\n---------------------------------------------\n'
      +"Calculation of Magnon Bands for DMI =  {:.3f}".format(DMI[1])
      +'\n--------------------------------------------\n')
##-------------------------------
##   Function for calculating the J_K 
##-------------------------------
##-- The J_K between the Site A and B
def J_K_AB(K) :
    AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        AB_FT.append(np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        AB_FT.append(np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    J_AB = np.sum(AB_FT)
    return J_AB
##-- The J_K between the Site A and A
def J_K_AA(K) :
    AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        AA_FT.append(mt.cos(phase+np.dot(K,R_J2[p])))
    J_AA = np.sum(AA_FT)
    return J_AA
##-- The J_K between the Site B and B
def J_K_BB(K) :
    BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        BB_FT.append(mt.cos(phase-np.dot(K,R_J2[p])))
    J_BB = np.sum(BB_FT)
    return J_BB
##---------------------------------------------------------------------------
##   Function for calculating the Hamiltonian and oocupation number(Strat)
##---------------------------------------------------------------------------
##----- Hamiltionian
def B_matrix(K0):
    Ha_B = np.zeros([2,2],dtype=complex) 
    ##############
    Ha_B[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_AA(K_BZ[K0]))-(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B
    ##############
    Ha_B[0,1] = 2*Spin*J_K_AB(K_BZ[K0])
    ##############
    Ha_B[1,0] = np.conjugate(Ha_B[0,1])
    ##############   
    Ha_B[1,1] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_BB(K_BZ[K0]))-(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B
    H_B = Ha_B
    return H_B #-- Hamiltonian of LSWT
##------------------------------------------------------------------------
##   Function for calculating the Hamiltonian and oocupation number (End)
##------------------------------------------------------------------------
##------------------------------------------------
##  Calculating the magnon-band for LSWT (Start)
##------------------------------------------------
##--
###--- EigenValues function for K points
E_A_B = []
psi_A = []
psi_A_star = []
E_O_B = []
psi_O = []
psi_O_star = []
###--- Wave function for K points
def PSI_A_O(kpoint):
    v_L_B, w_L_B = eigh(B_matrix(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(PSI_A_O, dataset)
        PSI_L = result
for i in range(K_BZ.shape[0]):
    E_A_B.append(PSI_L[i][0][0] )
    psi_A.append(PSI_L[i][1][0])
    psi_A_star.append(PSI_L[i][1][0].reshape(1,2).conj())
    ########
    E_O_B.append(PSI_L[i][0][1])
    psi_O.append(PSI_L[i][1][1])
    psi_O_star.append(PSI_L[i][1][1].reshape(1,2).conj())
E_BZ = np.zeros([N_M_S,K_BZ.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ[0,i] = E_A_B[i]
    E_BZ[1,i] = E_O_B[i]
en_cross = round(min(E_BZ[1,:]))
def Plot_Band_Structure3D(K,E): # E1 Energy for Acoustic mode & E2 for Optical mode
    figure = plt.figure(figsize=(6,5))
    ##--
    axis = figure.add_subplot(111, projection = '3d')
    x = K[:,0]
    y = K[:,1]
    en_ac = np.array([E[0,:],E[0,:]])
    en_op = np.array([E[1,:],E[1,:]])
    ##--
    axis.plot_wireframe(x, y, en_op, cstride=1, color='r', label='Optical')
    axis.plot_wireframe(x, y, en_ac, rstride=1, color='b', label='Acostic')
    axis.legend(loc='best', bbox_to_anchor=(0.35,0.77))
    axis.view_init(elev=4., azim=45)
    ##--
    axis.set_xlabel("$K_{x}$")
    axis.set_ylabel("$K_{y}$")
    axis.zaxis.set_rotate_label(False)  # disable automatic rotation
    axis.set_zlabel("Energy (meV)", rotation=90)
    axis.text2D(0.3, 0.79, "Magnon band structure", transform=axis.transAxes)
    ##-------------
    axis.set_xticks([round(min(x),1)+.1,0.0,round(max(x),1)-.1])
    axis.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [round(min(E[1,:])),round(min(E[1,:])),round(min(E[1,:])),
         round(min(E[1,:])),round(min(E[1,:])),round(min(E[1,:])),
         round(min(E[1,:]))]
             ])
    axis.scatter(x1, y1,z1 , color='c')
    plt.rcParams.update({'figure.max_open_warning': 0})
    figure.tight_layout()
    # make an PDF figure of a plot'Band_Structure_DMI_{}.pdf'.format(DMI[1])
    figure.savefig('Band_Structure_DMI_{}.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
##------------------------------------------
##   Function for Magnon-DOS Plots (Start)
##------------------------------------------
DOS_magnon = E_A_B
for i in E_O_B:
    DOS_magnon.append(i)
##-- now plot density of states
##-- now plot density of states
def P_DOS(E3):
    fig, ax = plt.subplots()
    ax.hist(E3,80,range=(np.min(E3),np.max(E3)))
    ax.set_ylim(0.0,100.0)
    # put title
    ax.set_title("Density of states")
    ax.set_xlabel("Band energy")
    ax.set_ylabel("Number of states")
    # make an PDF figure of a plot
    fig.tight_layout()
    fig.savefig("Inversion-symmetry_dos_CrBr3.pdf")
    plt.show()
    return
print('Spin = ' + str(Spin) + '  '+ str(chr(1115)) + '\n'+ '\n'
      +'Single-ion Anisotropy = ' + str(A) + ' (meV) '+'\n'+ '\n'
      +'g factor = '+str(g)+'\n'+ '\n'
      +'Boltzman Constant = ' + str(KB) + ' (meV/Kelvin) '+'\n'+ '\n'
      +'mu_{B} = ' + str(mu_B) + ' (meV/tesla) '+'\n'+ '\n'
      +'Magnetic feild B = ' + str(B) + ' (Tesla) '+'\n'+ '\n'
      +"J\u2081    = {}".format(J[0]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2081    = {}".format(DMI[0]) + ' (meV) '+'\n'+'\n'
      +"J\u2082    = {}".format(J[1]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2082    = {}".format(DMI[1]) + ' (meV) '+'\n'+'\n'
      +"J\u2083    = {}".format(J[2]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2083    = {}".format(DMI[2]) + ' (meV) '+'\n'+'\n'
      +"J\u0302    = {:0.2f}".format(J_tilda) + ' (meV) '+'\n'+'\n'
      +"E(\u0393)  = {:0.3f}".format(E_A_B[0])+ ' (meV) '+'\n'
      +'--------------------------------------------------')
Plot_Band_Structure3D(K_BZ,E_BZ)
#P_DOS(DOS_magnon)
##---------------------------
##-- Considering DMI! = 0.0
##--------------------------
DMI = [0.0,0.22,0.0]
##-- phase
if J[1] == 0 and DMI[1]== 0 :
    phase = 0.0
elif J[1] == 0 :
    phase =np.pi/2 # if J2 = 0.0
elif J[1] != 0 :
    phase = np.arctan(DMI[1]/(J[1]))
##--------------------------------
##  The Velocity Functions (Start)
##--------------------------------
#-- VX
##-- The velocity Site A and B
def VX_AB(K) :
    x_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        x_AB_FT.append(R_J1[q][0]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        x_AB_FT.append(R_J3[q][0]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_x_AB = np.sum(x_AB_FT)
    return V_x_AB
##-- The velocity Site A and A
def VX_AA(K) :
    x_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_AA_FT.append(R_J2[p][0]*mt.sin(phase-np.dot(K,R_J2[p])))
    V_x_AA = np.sum(x_AA_FT)
    return V_x_AA
##-- The velocity Site B and B
def VX_BB(K) :
    x_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_BB_FT.append(R_J2[p][0]*mt.sin(phase+np.dot(K,R_J2[p])))
    V_x_BB = np.sum(x_BB_FT)
    return V_x_BB
#-- Vy
##-- The velocity Site A and B
def VY_AB(K) :
    y_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        y_AB_FT.append(R_J1[q][1]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        y_AB_FT.append(R_J3[q][1]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_y_AB = np.sum(y_AB_FT)
    return V_y_AB
##-- The velocity Site A and A
def VY_AA(K) :
    y_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_AA_FT.append(R_J2[p][1]*mt.sin(phase-np.dot(K,R_J2[p])))
    V_y_AA = np.sum(y_AA_FT)
    return V_y_AA
##-- The velocity Site B and B
def VY_BB(K) :
    y_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_BB_FT.append(R_J2[p][1]*mt.sin(phase+np.dot(K,R_J2[p])))
    V_y_BB = np.sum(y_BB_FT)
    return V_y_BB
##--------------------------------
##  The Velocity Functions (End)
##--------------------------------
#-------------------------------
## Ploting Beery Phase Colorbar
##------------------------------
#---- Sampling 1BZ
K_BZ_Beery_plot_x_limit = []
for i in K_BZ_Beery_plot:
    if (i[0] <= round(K6[0],3)+.1 and i[0] >= 0) :
        K_BZ_Beery_plot_x_limit.append(i)
    if (i[0] >= round(K3[0],3)-.1 and i[0] < 0):
        K_BZ_Beery_plot_x_limit.append(i)
K_BZ_Beery_plot_y_limit = []
for i in K_BZ_Beery_plot_x_limit:
    if (i[1] <= round(K2[1],3)+.2 and i[1] > 0 ) :
        K_BZ_Beery_plot_y_limit.append(i)
    if (i[1] >= round(K5[1],3)-.2 and i[1] < 0):
        K_BZ_Beery_plot_y_limit.append(i)
K_BZ_Beery_plot = np.zeros([len(K_BZ_Beery_plot_y_limit),3])
for i in range(K_BZ_Beery_plot.shape[0]) :
    K_BZ_Beery_plot[i] = K_BZ_Beery_plot_y_limit[i]
##----- Hamiltionian
def B_matrix_Beery_plot(K0):
    Ha_B_Beery_plot = np.zeros([2,2],dtype=complex) 
    ##############
    Ha_B_Beery_plot[0,0] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_AA(K_BZ_Beery_plot[K0]))
                -(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B )
    ##############
    Ha_B_Beery_plot[0,1] = 2*Spin*J_K_AB(K_BZ_Beery_plot[K0])
    ##############
    Ha_B_Beery_plot[1,0] = np.conjugate(Ha_B_Beery_plot[0,1])
    ##############   
    Ha_B_Beery_plot[1,1] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_BB(K_BZ_Beery_plot[K0]))-(2*Spin*J_tilda)
                -(2*Spin*A)+(g*mu_B)*B)
    H_B_Beery_plot = Ha_B_Beery_plot
    return H_B_Beery_plot #-- Hamiltonian of LSWT
###--- EigenValues function for K points
E_A_B_Beery_plot = []
psi_A_Beery_plot = []
psi_A_star_Beery_plot = []
E_O_B_Beery_plot = []
psi_O_Beery_plot = []
psi_O_star_Beery_plot = []
###--- Wave function for K points
def PSI_A_O_Beery_plot(kpoint):
    v_L_B, w_L_B = eigh(B_matrix_Beery_plot(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result_Beery_plot = pool.map(PSI_A_O_Beery_plot, dataset)
        PSI_L_Beery_plot = result_Beery_plot
for i in range(K_BZ_Beery_plot.shape[0]):
    E_A_B_Beery_plot.append(PSI_L_Beery_plot[i][0][0])
    psi_A_Beery_plot.append(PSI_L_Beery_plot[i][1][0])
    psi_A_star_Beery_plot.append(PSI_L_Beery_plot[i][1][0].reshape(1,2).conj())
    ########
    E_O_B_Beery_plot.append(PSI_L_Beery_plot[i][0][1])
    psi_O_Beery_plot.append(PSI_L_Beery_plot[i][1][1])
    psi_O_star_Beery_plot.append(PSI_L_Beery_plot[i][1][1].reshape(1,2).conj())
E_BZ_Beery_plot = np.zeros([N_M_S,K_BZ_Beery_plot.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ_Beery_plot[0,i] = E_A_B_Beery_plot[i]
    E_BZ_Beery_plot[1,i] = E_O_B_Beery_plot[i]
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ_Beery_plot[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ_Beery_plot[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##----- Calculating the Chern Numbers
##-- Berry Function
def Beery_conections_plot(K0):
    beery_cof = 1./((E_O_B_Beery_plot[K0]-E_A_B_Beery_plot[K0])**2)
    B_C_1_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vy(K0),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vx(K0),psi_O_Beery_plot[K0])))
    B_C_2_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vx(K0).conj(),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vy(K0).conj(),psi_O_Beery_plot[K0])))
    B_C_3_plot = beery_cof*(B_C_1_plot+B_C_2_plot)
    return B_C_3_plot[0].imag
Be_CR = 0
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(Beery_conections_plot, dataset)
        Ber_CR_Beery_plot = np.array(result)
Be_CR_Beery_plot = [round(num, 1) for num in Ber_CR_Beery_plot]
def Be_phase_im_2D(Beery,K):
    # convert to arrays to make use of previous answer to similar question
    x = K[:,0]
    y = K[:,1]
    z = np.asarray(Beery)

    # Set up a regular grid of interpolation points
    nInterp = 400
    xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(), y.max(), nInterp)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate; there's also method='cubic' for 2-D data such as here
    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

    img = plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower', interpolation='nearest',
                     cmap='coolwarm',extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto') 
    plt.xlim(x.min()-.02,x.max()+.02)
    plt.ylim(y.min()-.03,y.max()+.03)
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")        
    cbar = plt.colorbar(img)
    cbar.ax.get_yaxis().labelpad = 7
    cbar.ax.set_ylabel('Beery cervature', rotation=90)
    plt.savefig('Berry_phase_DMI_{}_2D.pdf'.format(DMI[1]),dpi=1200,bbox_inches="tight")
    plt.show()
    return
#---------------------
def Be_phase_3D(Berry,K):
    X = K[:,0]
    Y = K[:,1]
    Z = np.array([(Berry),(Berry)])
    # Normalize to [0,1]
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.coolwarm(norm(Z))
    rcount, ccount, _ = colors.shape
    zlim = (0,0.1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    #--------------------------------
    ax.view_init(elev=25., azim=45)
    ax.set_xlabel("$K_{x}$")
    ax.set_ylabel("$K_{y}$")
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("Berry phase", rotation=90)
    ##-------------
    ax.set_xticks([round(min(X),1)+0.2,0.0,round(max(X),1)-0.2])
    ax.set_yticks([round(min(Y),1)+0.2,0.0,round(max(Y),1)-0.2])
    ##--------------
    fig.savefig('Berry_phase_DMI_{}_3D.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
#------------------------------------------------------------------
##-- Calculating Beery cervature & Chern number
print('---------------------------------------------------------------- '
   +'\nCalculating Beery cervature & Chern number (General Formulas)  :'
   +'\n---------------------------------------------------------------- '+'\n') 
###--- EigenValues function for K points
E_A_B = []
psi_A = []
psi_A_star = []
E_O_B = []
psi_O = []
psi_O_star = []
###--- Wave function for K points
def PSI_A_O(kpoint):
    v_L_B, w_L_B = eigh(B_matrix(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(PSI_A_O, dataset)
        PSI_L = result
for i in range(K_BZ.shape[0]):
    E_A_B.append(PSI_L[i][0][0])
    psi_A.append(PSI_L[i][1][0])
    psi_A_star.append(PSI_L[i][1][0].reshape(1,2).conj())
    ########
    E_O_B.append(PSI_L[i][0][1])
    psi_O.append(PSI_L[i][1][1])
    psi_O_star.append(PSI_L[i][1][1].reshape(1,2).conj())
E_BZ = np.zeros([N_M_S,K_BZ.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ[0,i] = E_A_B[i]
    E_BZ[1,i] = E_O_B[i]
##------------------------------------------
##   Function for Magnon-DOS Plots (Start)
##------------------------------------------
DOS = E_A_B
for i in E_O_B:
    DOS.append(i)
##-- now plot density of states
def P_DOS(E3):
    fig, ax = plt.subplots()
    ax.hist(E3,80,range=(np.min(E3),np.max(E3)))
    ax.set_ylim(0.0,100.0)
    # put title
    ax.set_title("Density of states")
    ax.set_xlabel("Band energy")
    ax.set_ylabel("Number of states")
    # make an PDF figure of a plot
    fig.tight_layout()
    #fig.savefig("Inversion-symmetry_dos_CrBr3.pdf")
    plt.show()
def Plot_Band_Structure3D(K,E): # E1 Energy for Acoustic mode & E2 for Optical mode
    figure = plt.figure(figsize=(6,5))
    axis = figure.add_subplot(111, projection = '3d')
    ##-----------------
    x = K[:,0]
    y = K[:,1]
    en_ac = np.array([E[0,:],E[0,:]])
    en_op = np.array([E[1,:],E[1,:]])
    ##----------------
    # axis.scatter(x, y, c='k', marker='*')
    axis.plot_wireframe(x, y, en_op, cstride=1, color='r', label='Optical')
    axis.plot_wireframe(x, y, en_ac, rstride=1, color='b', label='Acostic')
    axis.legend(loc='best', bbox_to_anchor=(0.35,0.77))
    axis.view_init(elev=4., azim=45)
    ##--------------
    axis.set_xlabel("$K_{x}$")
    axis.set_ylabel("$K_{y}$")
    axis.zaxis.set_rotate_label(False)  # disable automatic rotation
    axis.set_zlabel("Energy (meV)", rotation=90)
    axis.text2D(0.3, 0.79, "Magnon band structure", transform=axis.transAxes)
    ##-------------
    axis.set_xticks([round(min(x),1)+.1,0.0,round(max(x),1)-.1])
    axis.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ##---- 1BZ
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='m',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.scatter(x1, y1,z1 , color='k')
    ##--------
    figure.tight_layout()
    plt.rcParams.update({'figure.max_open_warning': 0})
    figure.savefig('Band_Structure_DMI_{}.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
print('\n----------------------------------------------\n'
      +"Calculation of Magnon Bands for DMI =  {:.3f}".format(DMI[1])
      +'\n----------------------------------------------\n')
Plot_Band_Structure3D(K_BZ,E_BZ)
#P_DOS(DOS)
if E_A_B[0] < 0.0 :
    print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n')
    print('   #########################################################################')
    print('   #Since the Energy at (\u0393) point is NEGATIVE, the calculation WILL STOP HERE#')
    print('   #########################################################################')
    print('------------------------------------------'+'\n')
    sys.exit()
print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n'
     +'E(Gap) = {:0.3f}'.format(min(E_BZ[1])-max(E_BZ[0]))+' (meV) '+'\n'
     )
print('------------------------------- '
   +'\nPlotting The Berry Phase (2D):'
   +'\n------------------------------- '+'\n') 
Be_phase_im_2D(Be_CR_Beery_plot,K_BZ_Beery_plot)
print('------------------------------- '
   +'\nPlotting The Berry Phase (3D):'
   +'\n------------------------------- '+'\n') 
Be_phase_3D(Be_CR_Beery_plot,K_BZ_Beery_plot)
##############################################
## Calculation Of Thermal Hall Conductivity
##############################################
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##-- Beery_conections_Optical
def B_C_O(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_O_star[K0],np.dot(Vy(K0),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vx(K0),psi_O[K0])))
    B_C_2 = ( np.dot(psi_O_star[K0],np.dot(Vx(K0).conj(),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vy(K0).conj(),psi_O[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_O, dataset)
        Ber_CR_O = np.array(result)
Be_CR_O = [round(num, 1) for num in Ber_CR_O]
Chern_Number_O = (sum(Be_CR_O)/(2*np.pi*K_BZ.shape[0]))
print(colored("C{}{} = {:.7f}".format(get_sub('n'),get_super('O'),Chern_Number_O)+'\n'+'\n', 'red'))
##-- Beery_conections_Acoustic
def B_C_A(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_A_star[K0],np.dot(Vy(K0),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vx(K0),psi_A[K0])))
    B_C_2 = ( np.dot(psi_A_star[K0],np.dot(Vx(K0).conj(),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vy(K0).conj(),psi_A[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_A, dataset)
        Ber_CR_A = np.array(result)
Be_CR_A = [round(num, 1) for num in Ber_CR_A]
Chern_Number_A = (sum(Be_CR_A)/(2*np.pi*K_BZ.shape[0]))
print(colored("C{}{} = {:.7f}".format(get_sub('n'),get_super('A'),Chern_Number_A)+'\n'+'\n', 'blue'))
#---------------------
############################################
##--    Temperature Range and steps
############################################
# intail_Temp = int(input("Enter Intail Temperature : ")) #
# Final_Temp  = int(input("Enter Final Temperature: ")) #
# steps_Temp  = int(input("Enter Steps of Temperature: ")) #
intail_Temp = 0
Final_Temp  = 31
steps_Temp  = 1
############################################
##-- Calculating The Ocupation Number & C2
############################################
import mpmath as MPM
MPM.mp.dps = 15; MPM.mp.pretty = True
print('---------------------------------------- '
   +'\nCalculating The ocupation number :'
   +'\n---------------------------------------- '+'\n')
##-- Bos-Einstin Distrbution
print(colored('-------------------\n'
     +'The Acoustic mode :'
     +'\n', 'blue'))
def nk_A(t,K): # Acostic Mode
    if t == 0.0 :
        oc_A = 0.0
    else:
        EXP_A = np.exp(np.float128((E_BZ[0,K])/(KB*t)))-1 # Exponential - 1
        oc_A = (1/EXP_A)
    ocup_A = oc_A
    return float("%.6f" %(ocup_A))
Temp =  [float("%.2f" %(i)) for i in np.arange(intail_Temp,Final_Temp,steps_Temp)]
##----- Ocupation number for Acustic mode
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_A,dataset)
n_k_A = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
##n_k_A = np.round_(n_k_A,2)
print(colored('Done.\n', 'blue'))
##----- Ocupation number for Optical mode
###
print(colored('------------------\n'
     +'The Optical mode :'
     +'\n', 'red'))
def nk_O(t,K): # Optical Mode
    if t == 0.0 :
        oc_O = 0
    else:
        EXP_O = np.exp(np.float128((E_BZ[1,K])/(KB*t)))-1 # Exponential - 1
        oc_O = (1/EXP_O)
    ocup_O = oc_O
    return float("%.6f" %(ocup_O))
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_O,dataset)
n_k_O = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
##n_k_O = np.round_(n_k_O,2)
print(colored('Done.\n', 'red'))
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(n_k_O[i,:],'r*')
#     axis[0].set_title('Ocupation for Optical mode')
#     axis[1].plot(n_k_A[i,:],'bo')
#     axis[1].set_title('Ocupation for Acustic mode')
#     plt.suptitle('Temperature_{}'.format(Temp[i]),fontsize=20)
#     fig.savefig("Ocupation_Temperature_{}.pdf".format(Temp[i]))
#     plt.show()
###############----------------------
##--- C2(nk_A)
print(colored('---------------------------------------- '
   +'\nCalculating The C2 for Acoustic mode :'
   +'\n---------------------------------------- '+'\n', 'blue'))
def C2_nk_A(x,y) :
    if n_k_A[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 = ( (1+n_k_A[x,y])*(np.log((1+n_k_A[x,y])/n_k_A[x,y])**2)
               -(np.log(n_k_A[x,y])**2)-(2*MPM.polylog(2, -n_k_A[x,y]))
              )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_A,dataset)
C2_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
##C2_A = np.round_(C2_A,2)
print(colored('Done.\n', 'blue'))
##--- C2(nk_O)
print(colored('---------------------------------------- '
   +'\nCalculating The C2 for Optical mode :'
   +'\n---------------------------------------- '+'\n', 'red'))
def C2_nk_O(x,y) :
    if n_k_O[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 =( (1+n_k_O[x,y])*(np.log((1+n_k_O[x,y])/n_k_O[x,y])**2)
             -(np.log(n_k_O[x,y])**2)-(2*MPM.polylog(2, -n_k_O[x,y]))
             )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_O,dataset)
C2_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
##C2_O = np.round_(C2_O,2)
print(colored('Done.\n', 'red'))
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_O[i,:],'r*')
#     axis[0].set_title('C2 for Optical mode')
#     axis[1].plot(C2_A[i,:],'bo')
#     axis[1].set_title('C2 for Acustic mode')
#     plt.suptitle('Temperature_{}'.format(Temp[i]),fontsize=20)
#     fig.savefig("C2_Temperature_{}.pdf".format(Temp[i]))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print(colored('---------------------------------------- '
   +'\nCalculating The C2 * BC Acoustic :'
   +'\n---------------------------------------- '+'\n', 'blue'))
##--- C2 * BC Acustic
def K_xy_A(t,K):
    C2BC_A = C2_A[t,K]*(Be_CR_A[K])
    return C2BC_A
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_A,dataset)
C2_BC_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
##C2_BC_A = np.round_(C2_BC_A,2)
print(colored('Done.\n', 'blue'))
print(colored('---------------------------------------- '
   +'\nCalculating The C2 * BC Optical :'
   +'\n---------------------------------------- '+'\n', 'red'))
##----- C2 * BC Optic
def K_xy_O(t,K):
    C2BC_O = C2_O[t,K]*(Be_CR_O[K])
    return C2BC_O
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_O,dataset)
C2_BC_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
##C2_BC_O = np.round_(C2_BC_O,2)
print(colored('Done.\n', 'red'))
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_BC_O[i,:],'r*')
#     axis[0].set_title('C2 * BC Optical mode')
#     axis[1].plot(C2_BC_A[i,:],'bo')
#     axis[1].set_title('C2 * BC Acustic mode')
#     plt.suptitle('Temperature_{}'.format(Temp[i]),fontsize=20)
#     fig.savefig("C2BC_Temperature_{}.pdf".format(Temp[i]))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print('---------------------------------------- '
   +'\nCalculating Thermal Hall Conductivity :'
   +'\n---------------------------------------- '+'\n')
K_XY_pos = []
K_XY_pos_A = []
K_XY_pos_O = []
for t in range(len(Temp)):
    start_itr_time = datetime.now()
    print('TEMPERATURE = {:.2f}'.format(Temp[t]) + '\n')
    K_XY_O = sum(C2_BC_O[t])/K_BZ.shape[0]
    K_XY_pos_O.append(K_XY_O)
    K_XY_A = sum(C2_BC_A[t])/K_BZ.shape[0]
    K_XY_pos_A.append(K_XY_A)
    cof = ((2*np.pi)**-2)*(Norm(np.cross(a,b))**-1)*(1.8)*Temp[t]
    K_XY_total = cof*(K_XY_O + K_XY_A)
    K_XY_pos.append(-K_XY_total) 
    ########################
    end_itr_time = datetime.now()
    print('K_XY_O = '+str(float("%.4f" %((K_XY_pos_O[t]))))+'\n'
         +'K_XY_A = '+str(float("%.4f" %((K_XY_pos_A[t]))))+'\n')
    print('Termal Hall Conductivity for CrBr\u2083 = '+str(float("%.4f" %((K_XY_pos[t]))))+'\n'
         +'\n'+'Duration of each Temperature step: {}'.format(end_itr_time - start_itr_time)
         +'\n---------------------------------')
#---- Ploting THC
def Plot_TH(KXY_p): 
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(KXY_p)) - 1
    end   = int(np.max(KXY_p)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    #ax.set_xlim(0,Temp[-2])
    ax.plot(Temp,KXY_p,'r', label='DMI = {}'.format(DMI[1]))
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    ax.set_title('Hall Conductivity')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa^{xy}x10^{-12}$(W/K)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig.savefig('THC_DMI_{}_T_{}.pdf'.format(DMI[1],Temp[t]),dpi=1000)
    plt.show()
    return
# print('--------------------------------\n'
#      +'Ploting THC for Acoustic mode :'
#      +'\n--------------------------------\n')
# Plot_TH(K_XY_pos_A)
# print('Done.\n')
# print('--------------------------------\n'
#      +'Ploting THC for Optical mode :'
#      +'\n--------------------------------\n')
# Plot_TH(K_XY_pos_O)
# print('Done.\n')
print('---------------\n'
     +'Ploting THC :'
   +'\n---------------\n')
Plot_TH(K_XY_pos)
###########################################################
##-- Considering DMI x -1
###########################################################
DMI = [0.0,-0.22,0.0]
print('\n------------\n'
      +'DMI x -1 : '
      +'\n-----------\n')
##-- phase
if J[1] == 0 and DMI[1]== 0 :
    phase = 0.0
elif J[1] == 0 :
    phase =np.pi/2 # if J2 = 0.0
elif J[1] != 0 :
    phase = np.arctan(DMI[1]/(J[1]))
##--------------------------------
##  The Velocity Functions (Start)
##--------------------------------
#-- VX
##-- The velocity Site A and B
def VX_AB(K) :
    x_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        x_AB_FT.append(R_J1[q][0]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        x_AB_FT.append(R_J3[q][0]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_x_AB = np.sum(x_AB_FT)
    return V_x_AB
##-- The velocity Site A and A
def VX_AA(K) :
    x_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_AA_FT.append(R_J2[p][0]*mt.sin(phase-np.dot(K,R_J2[p])))
    V_x_AA = np.sum(x_AA_FT)
    return V_x_AA
##-- The velocity Site B and B
def VX_BB(K) :
    x_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_BB_FT.append(R_J2[p][0]*mt.sin(phase+np.dot(K,R_J2[p])))
    V_x_BB = np.sum(x_BB_FT)
    return V_x_BB
#-- Vy
##-- The velocity Site A and B
def VY_AB(K) :
    y_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        y_AB_FT.append(R_J1[q][1]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        y_AB_FT.append(R_J3[q][1]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_y_AB = np.sum(y_AB_FT)
    return V_y_AB
##-- The velocity Site A and A
def VY_AA(K) :
    y_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_AA_FT.append(R_J2[p][1]*mt.sin(phase-np.dot(K,R_J2[p])))
    V_y_AA = np.sum(y_AA_FT)
    return V_y_AA
##-- The velocity Site B and B
def VY_BB(K) :
    y_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_BB_FT.append(R_J2[p][1]*mt.sin(phase+np.dot(K,R_J2[p])))
    V_y_BB = np.sum(y_BB_FT)
    return V_y_BB
##--------------------------------
##  The Velocity Functions (End)
##--------------------------------
#-------------------------------
## Ploting Beery Phase Colorbar
##------------------------------
#---- Sampling 1BZ
K_BZ_Beery_plot_x_limit = []
for i in K_BZ_Beery_plot:
    if (i[0] <= round(K6[0],3)+.1 and i[0] >= 0) :
        K_BZ_Beery_plot_x_limit.append(i)
    if (i[0] >= round(K3[0],3)-.1 and i[0] < 0):
        K_BZ_Beery_plot_x_limit.append(i)
K_BZ_Beery_plot_y_limit = []
for i in K_BZ_Beery_plot_x_limit:
    if (i[1] <= round(K2[1],3)+.2 and i[1] > 0 ) :
        K_BZ_Beery_plot_y_limit.append(i)
    if (i[1] >= round(K5[1],3)-.2 and i[1] < 0):
        K_BZ_Beery_plot_y_limit.append(i)
K_BZ_Beery_plot = np.zeros([len(K_BZ_Beery_plot_y_limit),3])
for i in range(K_BZ_Beery_plot.shape[0]) :
    K_BZ_Beery_plot[i] = K_BZ_Beery_plot_y_limit[i]
##----- Hamiltionian
def B_matrix_Beery_plot(K0):
    Ha_B_Beery_plot = np.zeros([2,2],dtype=complex) 
    ##############
    Ha_B_Beery_plot[0,0] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_AA(K_BZ_Beery_plot[K0]))
                -(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B )
    ##############
    Ha_B_Beery_plot[0,1] = 2*Spin*J_K_AB(K_BZ_Beery_plot[K0])
    ##############
    Ha_B_Beery_plot[1,0] = np.conjugate(Ha_B_Beery_plot[0,1])
    ##############   
    Ha_B_Beery_plot[1,1] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_BB(K_BZ_Beery_plot[K0]))-(2*Spin*J_tilda)
                -(2*Spin*A)+(g*mu_B)*B)
    H_B_Beery_plot = Ha_B_Beery_plot
    return H_B_Beery_plot #-- Hamiltonian of LSWT
###--- EigenValues function for K points
E_A_B_Beery_plot = []
psi_A_Beery_plot = []
psi_A_star_Beery_plot = []
E_O_B_Beery_plot = []
psi_O_Beery_plot = []
psi_O_star_Beery_plot = []
###--- Wave function for K points
def PSI_A_O_Beery_plot(kpoint):
    v_L_B, w_L_B = eigh(B_matrix_Beery_plot(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result_Beery_plot = pool.map(PSI_A_O_Beery_plot, dataset)
        PSI_L_Beery_plot = result_Beery_plot
for i in range(K_BZ_Beery_plot.shape[0]):
    E_A_B_Beery_plot.append(PSI_L_Beery_plot[i][0][0])
    psi_A_Beery_plot.append(PSI_L_Beery_plot[i][1][0])
    psi_A_star_Beery_plot.append(PSI_L_Beery_plot[i][1][0].reshape(1,2).conj())
    ########
    E_O_B_Beery_plot.append(PSI_L_Beery_plot[i][0][1])
    psi_O_Beery_plot.append(PSI_L_Beery_plot[i][1][1])
    psi_O_star_Beery_plot.append(PSI_L_Beery_plot[i][1][1].reshape(1,2).conj())
E_BZ_Beery_plot = np.zeros([N_M_S,K_BZ_Beery_plot.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ_Beery_plot[0,i] = E_A_B_Beery_plot[i]
    E_BZ_Beery_plot[1,i] = E_O_B_Beery_plot[i]
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ_Beery_plot[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ_Beery_plot[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##----- Calculating the Chern Numbers
##-- Berry Function
def Beery_conections_plot(K0):
    beery_cof = 1./((E_O_B_Beery_plot[K0]-E_A_B_Beery_plot[K0])**2)
    B_C_1_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vy(K0),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vx(K0),psi_O_Beery_plot[K0])))
    B_C_2_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vx(K0).conj(),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vy(K0).conj(),psi_O_Beery_plot[K0])))
    B_C_3_plot = beery_cof*(B_C_1_plot+B_C_2_plot)
    return B_C_3_plot[0].imag
Be_CR = 0
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(Beery_conections_plot, dataset)
        Ber_CR_Beery_plot = np.array(result)
Be_CR_Beery_plot = [round(num, 1) for num in Ber_CR_Beery_plot]
def Be_phase_im_2D(Beery,K):
    # convert to arrays to make use of previous answer to similar question
    x = K[:,0]
    y = K[:,1]
    z = np.asarray(Beery)

    # Set up a regular grid of interpolation points
    nInterp = 400
    xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(), y.max(), nInterp)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate; there's also method='cubic' for 2-D data such as here
    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

    img = plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower', interpolation='nearest',
                     cmap='coolwarm',extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto') 
    plt.xlim(x.min()-.02,x.max()+.02)
    plt.ylim(y.min()-.03,y.max()+.03)
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")        
    cbar = plt.colorbar(img)
    cbar.ax.get_yaxis().labelpad = 7
    cbar.ax.set_ylabel('Beery cervature', rotation=90)
    plt.savefig('Berry_phase_DMII_{}_2D.pdf'.format(DMI[1]),dpi=1200,bbox_inches="tight")
    plt.show()
    return
#---------------------
def Be_phase_3D(Berry,K):
    X = K[:,0]
    Y = K[:,1]
    Z = np.array([(Berry),(Berry)])
    # Normalize to [0,1]
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.coolwarm(norm(Z))
    rcount, ccount, _ = colors.shape
    zlim = (0,0.1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    #--------------------------------
    ax.view_init(elev=25., azim=45)
    ax.set_xlabel("$K_{x}$")
    ax.set_ylabel("$K_{y}$")
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("Berry phase", rotation=90)
    ##-------------
    ax.set_xticks([round(min(X),1)+0.2,0.0,round(max(X),1)-0.2])
    ax.set_yticks([round(min(Y),1)+0.2,0.0,round(max(Y),1)-0.2])
    ##--------------
    fig.savefig('Berry_phase_DMII_{}_3D.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
#------------------------------------------------------------------
##-- Calculating Beery cervature & Chern number
print('---------------------------------------------------------------- '
   +'\nCalculating Beery cervature & Chern number (General Formulas)  :'
   +'\n---------------------------------------------------------------- '+'\n') 
###--- EigenValues function for K points
E_A_B = []
psi_A = []
psi_A_star = []
E_O_B = []
psi_O = []
psi_O_star = []
###--- Wave function for K points
def PSI_A_O(kpoint):
    v_L_B, w_L_B = eigh(B_matrix(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(PSI_A_O, dataset)
        PSI_L = result
for i in range(K_BZ.shape[0]):
    E_A_B.append(PSI_L[i][0][0])
    psi_A.append(PSI_L[i][1][0])
    psi_A_star.append(PSI_L[i][1][0].reshape(1,2).conj())
    ########
    E_O_B.append(PSI_L[i][0][1])
    psi_O.append(PSI_L[i][1][1])
    psi_O_star.append(PSI_L[i][1][1].reshape(1,2).conj())
E_BZ = np.zeros([N_M_S,K_BZ.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ[0,i] = E_A_B[i]
    E_BZ[1,i] = E_O_B[i]
##------------------------------------------
##   Function for Magnon-DOS Plots (Start)
##------------------------------------------
DOS = E_A_B
for i in E_O_B:
    DOS.append(i)
##-- now plot density of states
def P_DOS(E3):
    fig, ax = plt.subplots()
    ax.hist(E3,80,range=(np.min(E3),np.max(E3)))
    ax.set_ylim(0.0,100.0)
    # put title
    ax.set_title("Density of states")
    ax.set_xlabel("Band energy")
    ax.set_ylabel("Number of states")
    # make an PDF figure of a plot
    fig.tight_layout()
    #fig.savefig("Inversion-symmetry_dos_CrBr3.pdf")
    plt.show()
def Plot_Band_Structure3D(K,E): # E1 Energy for Acoustic mode & E2 for Optical mode
    figure = plt.figure(figsize=(6,5))
    axis = figure.add_subplot(111, projection = '3d')
    ##-----------------
    x = K[:,0]
    y = K[:,1]
    en_ac = np.array([E[0,:],E[0,:]])
    en_op = np.array([E[1,:],E[1,:]])
    ##----------------
    # axis.scatter(x, y, c='k', marker='*')
    axis.plot_wireframe(x, y, en_op, cstride=1, color='r', label='Optical')
    axis.plot_wireframe(x, y, en_ac, rstride=1, color='b', label='Acostic')
    axis.legend(loc='best', bbox_to_anchor=(0.35,0.77))
    axis.view_init(elev=4., azim=45)
    ##--------------
    axis.set_xlabel("$K_{x}$")
    axis.set_ylabel("$K_{y}$")
    axis.zaxis.set_rotate_label(False)  # disable automatic rotation
    axis.set_zlabel("Energy (meV)", rotation=90)
    axis.text2D(0.3, 0.79, "Magnon band structure", transform=axis.transAxes)
    ##-------------
    axis.set_xticks([round(min(x),1)+.1,0.0,round(max(x),1)-.1])
    axis.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ##---- 1BZ
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.scatter(x1, y1,z1 , color='k')
    ##--------
    figure.tight_layout()
    plt.rcParams.update({'figure.max_open_warning': 0})
    figure.savefig('Band_Structure_DMI_{}.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
print('\n----------------------------------------------\n'
      +"Calculation of Magnon Bands for DMI =  {:.3f}".format(DMI[1])
      +'\n----------------------------------------------\n')
Plot_Band_Structure3D(K_BZ,E_BZ)
#P_DOS(DOS)
if E_A_B[0] < 0.0 :
    print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n')
    print('   #########################################################################')
    print('   #Since the Energy at (\u0393) point is NEGATIVE, the calculation WILL STOP HERE#')
    print('   #########################################################################')
    print('------------------------------------------'+'\n')
    sys.exit()
print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n'
     +'E(Gap) = {:0.3f}'.format(min(E_BZ[1])-max(E_BZ[0]))+' (meV) '+'\n'
     )
print('------------------------------- '
   +'\nPlotting The Berry Phase (2D):'
   +'\n------------------------------- '+'\n') 
Be_phase_im_2D(Be_CR_Beery_plot,K_BZ_Beery_plot)
print('------------------------------- '
   +'\nPlotting The Berry Phase (3D):'
   +'\n------------------------------- '+'\n') 
Be_phase_3D(Be_CR_Beery_plot,K_BZ_Beery_plot)
##############################################
## Calculation Of Thermal Hall Conductivity
##############################################
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##-- Beery_conections_Optical
def B_C_O(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_O_star[K0],np.dot(Vy(K0),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vx(K0),psi_O[K0])))
    B_C_2 = ( np.dot(psi_O_star[K0],np.dot(Vx(K0).conj(),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vy(K0).conj(),psi_O[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_O, dataset)
        Ber_CR_O = np.array(result)
Be_CR_O = [round(num, 1) for num in Ber_CR_O]
Chern_Number_O = (sum(Be_CR_O)/(2*np.pi*K_BZ.shape[0]))
print(colored("C{}{} = {:.7f}".format(get_sub('n'),get_super('O'),Chern_Number_O)+'\n'+'\n', 'red'))
##-- Beery_conections_Acoustic
def B_C_A(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_A_star[K0],np.dot(Vy(K0),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vx(K0),psi_A[K0])))
    B_C_2 = ( np.dot(psi_A_star[K0],np.dot(Vx(K0).conj(),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vy(K0).conj(),psi_A[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_A, dataset)
        Ber_CR_A = np.array(result)
Be_CR_A = [round(num, 1) for num in Ber_CR_A]
Chern_Number_A = (sum(Be_CR_A)/(2*np.pi*K_BZ.shape[0]))
print(colored("C{}{} = {:.7f}".format(get_sub('n'),get_super('A'),Chern_Number_A)+'\n'+'\n', 'blue'))
############################################
##-- Calculating The Ocupation Number & C2
############################################
import mpmath as MPM
MPM.mp.dps = 15; MPM.mp.pretty = True
print('---------------------------------------- '
   +'\nCalculating The ocupation number :'
   +'\n---------------------------------------- '+'\n')
##-- Bos-Einstin Distrbution
def nk_A(t,K): # Acostic Mode
    if t == 0.0 :
        oc_A = 0.0
    else:
        EXP_A = np.exp(np.float128((E_BZ[0,K])/(KB*t)))-1 # Exponential - 1
        oc_A = (1/EXP_A)
    ocup_A = oc_A
    return float("%.6f" %(ocup_A))
##----------------------------
Temp =  [float("%.2f" %(i)) for i in np.arange(intail_Temp,Final_Temp,steps_Temp)]
##----- Ocupation number for Acustic mode
print(colored('---------------------------------------- '
   +'\nThe Acoustic mode :'
   +'\n', 'blue'))
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_A,dataset)
n_k_A = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
print(colored('Done.\n', 'blue'))
##----- Ocupation number for Optical mode
###
def nk_O(t,K): # Optical Mode
    if t == 0.0 :
        oc_O = 0
    else:
        EXP_O = np.exp(np.float128((E_BZ[1,K])/(KB*t)))-1 # Exponential - 1
        oc_O = (1/EXP_O)
    ocup_O = oc_O
    return float("%.6f" %(ocup_O))
print(colored('---------------------------------------- '
   +'\nThe Optical mode :'
   +'\n', 'red'))
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_O,dataset)
n_k_O = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
print(colored('Done.\n', 'red'))
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(n_k_O[i,:],'r*')
#     axis[0].set_title('Ocupation for Optical mode')
#     axis[1].plot(n_k_A[i,:],'bo')
#     axis[1].set_title('Ocupation for Acustic mode')
#     #fig.savefig("Temperature_{}.pdf".format(i))
#     plt.show()
###############----------------------
##--- C2(nk_A)
print(colored('---------------------------------------- '
   +'\nCalculating The C2 for Acoustic mode :'
   +'\n---------------------------------------- '+'\n', 'blue'))
def C2_nk_A(x,y) :
    if n_k_A[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 = ( (1+n_k_A[x,y])*(np.log((1+n_k_A[x,y])/n_k_A[x,y])**2)
               -(np.log(n_k_A[x,y])**2)-(2*MPM.polylog(2, -n_k_A[x,y]))
              )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_A,dataset)
C2_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
print(colored('Done.\n', 'blue'))
##--- C2(nk_O)
print(colored('---------------------------------------- '
   +'\nCalculating The C2 for Optical mode :'
   +'\n---------------------------------------- '+'\n', 'red'))
def C2_nk_O(x,y) :
    if n_k_O[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 =( (1+n_k_O[x,y])*(np.log((1+n_k_O[x,y])/n_k_O[x,y])**2)
             -(np.log(n_k_O[x,y])**2)-(2*MPM.polylog(2, -n_k_O[x,y]))
             )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_O,dataset)
C2_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
print(colored('Done.\n', 'red'))
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_O[i,:],'r*')
#     axis[0].set_title('C2 for Optical mode')
#     axis[1].plot(C2_A[i,:],'bo')
#     axis[1].set_title('C2 for Acustic mode')
#     #fig.savefig("Temperature_{}.pdf".format(i))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print(colored('---------------------------------------- '
   +'\nCalculating The C2 * BC Acoustic :'
   +'\n---------------------------------------- '+'\n', 'blue'))
##--- C2 * BC Acustic
def K_xy_A(t,K):
    C2BC_A = C2_A[t,K]*(Be_CR_A[K])
    return C2BC_A
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_A,dataset)
C2_BC_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
print(colored('Done.\n', 'blue'))
print(colored('---------------------------------------- '
   +'\nCalculating The C2 * BC Optical :'
   +'\n---------------------------------------- '+'\n', 'red'))
##----- C2 * BC Optic
def K_xy_O(t,K):
    C2BC_O = C2_O[t,K]*(Be_CR_O[K])
    return C2BC_O
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_O,dataset)
C2_BC_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
print(colored('Done.\n', 'red'))
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_BC_O[i,:],'r*')
#     axis[0].set_title('C2 * BC Optical mode')
#     axis[1].plot(C2_BC_A[i,:],'bo')
#     axis[1].set_title('C2 * BC Acustic mode')
#     #fig.savefig("C2BC_Temperature_{}.pdf".format(i))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print('---------------------------------------- '
   +'\nCalculating Thermal Hall Conductivity :'
   +'\n---------------------------------------- '+'\n')
K_XY_neg = []
K_XY_neg_A = []
K_XY_neg_O = []
for t in range(len(Temp)):
    start_itr_time = datetime.now()
    print('TEMPERATURE = {:.2f}'.format(Temp[t]) + '\n')
    K_XY_O = sum(C2_BC_O[t])/K_BZ.shape[0]
    K_XY_neg_O.append(K_XY_O)
    K_XY_A = sum(C2_BC_A[t])/K_BZ.shape[0]
    K_XY_neg_A.append(K_XY_A)
    cof = ((2*np.pi)**-2)*(Norm(np.cross(a,b))**-1)*(1.8)*Temp[t]
    K_XY_total = cof*(K_XY_O + K_XY_A)
    K_XY_neg.append(-K_XY_total) 
    ########################
    end_itr_time = datetime.now()
    print('K_XY_O = '+str(float("%.4f" %((K_XY_neg_O[t]))))+'\n'
         +'K_XY_A = '+str(float("%.4f" %((K_XY_neg_A[t]))))+'\n')
    print('Termal Hall Conductivity for CrBr\u2083 = '+str(float("%.4f" %((K_XY_neg[t]))))+'\n'
         +'\n'+'Duration of each Temperature step: {}'.format(end_itr_time - start_itr_time)
         +'\n---------------------------------')
#---- Ploting THC
def Plot_TH_N(KXY_n): 
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(KXY_n)) - 1
    end   = int(np.max(KXY_n)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    #ax.set_xlim(0,Temp[-2])
    ax.plot(Temp,KXY_n,'b', label='DMI = {}'.format(DMI[1]))
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    ax.set_title('Hall Conductivity')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa^{xy}x10^{-12}$(W/K)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    fig.savefig('THC_DMI_{}_T_{}.pdf'.format(DMI[1],Temp[t]),dpi=1000)
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.show()
    return
Plot_TH_N(K_XY_neg)
print('-----------------------\n'
     +'Ploting THC for Both :'
   +'\n-----------------------\n')
#---- Ploting THC
def Plot_TH_T(KXY_p,KXY_n): 
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(KXY_p)) - 1
    end   = int(np.max(KXY_p)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    #ax.set_xlim(0,Temp[-2])
    ax.plot(Temp,KXY_p,'r', label='DMI = {}'.format(-DMI[1]))
    ax.plot(Temp,KXY_n,'b', label='DMI = {}'.format(DMI[1]))
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    ax.set_title('Hall Conductivity')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa^{xy}x10^{-12}$(W/K)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    fig.savefig('THC_both_T_{}.pdf'.format(Temp[t]),dpi=1000)
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.show()
    return
Plot_TH_T(K_XY_pos,K_XY_neg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


##---------------------------
##-- Considering DMI < 0.0
##--------------------------
DMI = [0.0,-0.22,0.0]
##-- phase
if J[1] == 0 and DMI[1]== 0 :
    phase = 0.0
elif J[1] == 0 :
    phase =np.pi/2 # if J2 = 0.0
elif J[1] != 0 :
    phase = np.arctan(DMI[1]/(J[1]))
##--------------------------------
##  The Velocity Functions (Start)
##--------------------------------
#-- VX
##-- The velocity Site A and B
def VX_AB(K) :
    x_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        x_AB_FT.append(R_J1[q][0]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        x_AB_FT.append(R_J3[q][0]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_x_AB = np.sum(x_AB_FT)
    return V_x_AB
##-- The velocity Site A and A
def VX_AA(K) :
    x_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_AA_FT.append(R_J2[p][0]*np.sin(phase+np.dot(K,R_J2[p])))
    V_x_AA = np.sum(x_AA_FT)
    return V_x_AA
##-- The velocity Site B and B
def VX_BB(K) :
    x_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_BB_FT.append(R_J2[p][0]*np.sin(phase-np.dot(K,R_J2[p])))
    V_x_BB = np.sum(x_BB_FT)
    return V_x_BB
#-- Vy
##-- The velocity Site A and B
def VY_AB(K) :
    y_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        y_AB_FT.append(R_J1[q][1]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        y_AB_FT.append(R_J3[q][1]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_y_AB = np.sum(y_AB_FT)
    return V_y_AB
##-- The velocity Site A and A
def VY_AA(K) :
    y_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_AA_FT.append(R_J2[p][1]*np.sin(phase+np.dot(K,R_J2[p])))
    V_y_AA = np.sum(y_AA_FT)
    return V_y_AA
##-- The velocity Site B and B
def VY_BB(K) :
    y_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_BB_FT.append(R_J2[p][1]*np.sin(phase-np.dot(K,R_J2[p])))
    V_y_BB = np.sum(y_BB_FT)
    return V_y_BB
##--------------------------------
##  The Velocity Functions (End)
##--------------------------------
#-------------------------------
## Ploting Beery Phase Colorbar
##------------------------------
#---- Sampling 1BZ
K_BZ_Beery_plot_x_limit = []
for i in K_BZ_Beery_plot:
    if (i[0] <= round(K6[0],3)+.1 and i[0] >= 0) :
        K_BZ_Beery_plot_x_limit.append(i)
    if (i[0] >= round(K3[0],3)-.1 and i[0] < 0):
        K_BZ_Beery_plot_x_limit.append(i)
K_BZ_Beery_plot_y_limit = []
for i in K_BZ_Beery_plot_x_limit:
    if (i[1] <= round(K2[1],3)+.2 and i[1] > 0 ) :
        K_BZ_Beery_plot_y_limit.append(i)
    if (i[1] >= round(K5[1],3)-.2 and i[1] < 0):
        K_BZ_Beery_plot_y_limit.append(i)
K_BZ_Beery_plot = np.zeros([len(K_BZ_Beery_plot_y_limit),3])
for i in range(K_BZ_Beery_plot.shape[0]) :
    K_BZ_Beery_plot[i] = K_BZ_Beery_plot_y_limit[i]
##----- Hamiltionian
def B_matrix_Beery_plot(K0):
    Ha_B_Beery_plot = np.zeros([2,2],dtype=complex) 
    ##############
    Ha_B_Beery_plot[0,0] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_AA(K_BZ_Beery_plot[K0]))
                -(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B )
    ##############
    Ha_B_Beery_plot[0,1] = 2*Spin*J_K_AB(K_BZ_Beery_plot[K0])
    ##############
    Ha_B_Beery_plot[1,0] = np.conjugate(Ha_B_Beery_plot[0,1])
    ##############   
    Ha_B_Beery_plot[1,1] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_BB(K_BZ_Beery_plot[K0]))-(2*Spin*J_tilda)
                -(2*Spin*A)+(g*mu_B)*B)
    H_B_Beery_plot = Ha_B_Beery_plot
    return H_B_Beery_plot #-- Hamiltonian of LSWT
###--- EigenValues function for K points
E_A_B_Beery_plot = []
psi_A_Beery_plot = []
psi_A_star_Beery_plot = []
E_O_B_Beery_plot = []
psi_O_Beery_plot = []
psi_O_star_Beery_plot = []
###--- Wave function for K points
def PSI_A_O_Beery_plot(kpoint):
    v_L_B, w_L_B = eigh(B_matrix_Beery_plot(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result_Beery_plot = pool.map(PSI_A_O_Beery_plot, dataset)
        PSI_L_Beery_plot = result_Beery_plot
for i in range(K_BZ_Beery_plot.shape[0]):
    E_A_B_Beery_plot.append(PSI_L_Beery_plot[i][0][0])
    psi_A_Beery_plot.append(PSI_L_Beery_plot[i][1][0])
    psi_A_star_Beery_plot.append(PSI_L_Beery_plot[i][1][0].reshape(1,2).conj())
    ########
    E_O_B_Beery_plot.append(PSI_L_Beery_plot[i][0][1])
    psi_O_Beery_plot.append(PSI_L_Beery_plot[i][1][1])
    psi_O_star_Beery_plot.append(PSI_L_Beery_plot[i][1][1].reshape(1,2).conj())
E_BZ_Beery_plot = np.zeros([N_M_S,K_BZ_Beery_plot.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ_Beery_plot[0,i] = E_A_B_Beery_plot[i]
    E_BZ_Beery_plot[1,i] = E_O_B_Beery_plot[i]
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ_Beery_plot[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ_Beery_plot[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##----- Calculating the Chern Numbers
##-- Berry Function
def Beery_conections_plot(K0):
    beery_cof = 1./((E_O_B_Beery_plot[K0]-E_A_B_Beery_plot[K0])**2)
    B_C_1_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vy(K0),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vx(K0),psi_O_Beery_plot[K0])))
    B_C_2_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vx(K0).conj(),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vy(K0).conj(),psi_O_Beery_plot[K0])))
    B_C_3_plot = beery_cof*(B_C_1_plot+B_C_2_plot)
    return B_C_3_plot[0].imag
Be_CR = 0
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(Beery_conections_plot, dataset)
        Ber_CR_Beery_plot = np.array(result)
Be_CR_Beery_plot = [round(num, 1) for num in Ber_CR_Beery_plot]
def Be_phase_im_2D(Beery,K):
    # convert to arrays to make use of previous answer to similar question
    x = K[:,0]
    y = K[:,1]
    z = np.asarray(Beery)

    # Set up a regular grid of interpolation points
    nInterp = 400
    xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(), y.max(), nInterp)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate; there's also method='cubic' for 2-D data such as here
    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

    img = plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower', interpolation='nearest',
                     cmap='coolwarm',extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto') 
    plt.xlim(x.min()-.02,x.max()+.02)
    plt.ylim(y.min()-.03,y.max()+.03)
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")        
    cbar = plt.colorbar(img)
    cbar.ax.get_yaxis().labelpad = 7
    cbar.ax.set_ylabel('Beery cervature', rotation=90)
    #plt.savefig('Berry_phase_positive-DMI_2D.pdf',dpi=1200,bbox_inches="tight")
    plt.show()
    return
#---------------------
def Be_phase_3D(Berry,K):
    X = K[:,0]
    Y = K[:,1]
    Z = np.array([(Berry),(Berry)])
    # Normalize to [0,1]
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.coolwarm(norm(Z))
    rcount, ccount, _ = colors.shape
    zlim = (0,0.1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    #--------------------------------
    ax.view_init(elev=25., azim=45)
    ax.set_xlabel("$K_{x}$")
    ax.set_ylabel("$K_{y}$")
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("Berry phase", rotation=90)
    ##-------------
    ax.set_xticks([round(min(X),1)+0.2,0.0,round(max(X),1)-0.2])
    ax.set_yticks([round(min(Y),1)+0.2,0.0,round(max(Y),1)-0.2])
    ##-------------
    fig.savefig("Berry_phase_negative-DMI_3D.pdf",dpi=2**20)
    plt.show()
    return
#------------------------------------------------------------------
##-- Calculating Beery cervature & Chern number
print('------------------------------------------------------------------------ '
   +'\nCalculating Beery cervature & Chern number (General Formulas)  :'
   +'\n------------------------------------------------------------------------ '+'\n') 
###--- EigenValues function for K points
E_A_B = []
psi_A = []
psi_A_star = []
E_O_B = []
psi_O = []
psi_O_star = []
###--- Wave function for K points
def PSI_A_O(kpoint):
    v_L_B, w_L_B = eigh(B_matrix(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(PSI_A_O, dataset)
        PSI_L = result
for i in range(K_BZ.shape[0]):
    E_A_B.append(PSI_L[i][0][0])
    psi_A.append(PSI_L[i][1][0])
    psi_A_star.append(PSI_L[i][1][0].reshape(1,2).conj())
    ########
    E_O_B.append(PSI_L[i][0][1])
    psi_O.append(PSI_L[i][1][1])
    psi_O_star.append(PSI_L[i][1][1].reshape(1,2).conj())
E_BZ = np.zeros([N_M_S,K_BZ.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ[0,i] = E_A_B[i]
    E_BZ[1,i] = E_O_B[i]
##------------------------------------------
##   Function for Magnon-DOS Plots (Start)
##------------------------------------------
DOS = E_A_B
for i in E_O_B:
    DOS.append(i)
##-- now plot density of states
def P_DOS(E3):
    fig, ax = plt.subplots()
    ax.hist(E3,80,range=(np.min(E3),np.max(E3)))
    ax.set_ylim(0.0,100.0)
    # put title
    ax.set_title("Density of states")
    ax.set_xlabel("Band energy")
    ax.set_ylabel("Number of states")
    # make an PDF figure of a plot
    fig.tight_layout()
    #fig.savefig("Inversion-symmetry_dos_CrBr3.pdf")
    plt.show()
def Plot_Band_Structure3D(K,E): # E1 Energy for Acoustic mode & E2 for Optical mode
    figure = plt.figure(figsize=(6,5))
    axis = figure.add_subplot(111, projection = '3d')
    ##-----------------
    x = K[:,0]
    y = K[:,1]
    en_ac = np.array([E[0,:],E[0,:]])
    en_op = np.array([E[1,:],E[1,:]])
    ##----------------
    # axis.scatter(x, y, c='k', marker='*')
    axis.plot_wireframe(x, y, en_op, cstride=1, color='r', label='Optical')
    axis.plot_wireframe(x, y, en_ac, rstride=1, color='b', label='Acostic')
    axis.legend(loc='best', bbox_to_anchor=(0.35,0.77))
    axis.view_init(elev=4., azim=45)
    ##--------------
    axis.set_xlabel("$K_{x}$")
    axis.set_ylabel("$K_{y}$")
    axis.zaxis.set_rotate_label(False)  # disable automatic rotation
    axis.set_zlabel("Energy (meV)", rotation=90)
    axis.text2D(0.3, 0.79, "Magnon band structure", transform=axis.transAxes)
    ##-------------
    axis.set_xticks([round(min(x),1)+.1,0.0,round(max(x),1)-.1])
    axis.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ##---- 1BZ
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.scatter(x1, y1,z1 , color='k')
    ##--------
    figure.tight_layout()
    plt.rcParams.update({'figure.max_open_warning': 0})
    figure.savefig("Band_Structure_Gapped_negative.pdf",dpi=2**20)
    plt.show()
    return
print('\n----------------------------------------------------------\n'
      +"Calculation of Magnon Bands for DMI =  {:.3f}".format(DMI[1])
      +'\n---------------------------------------------------------\n')
print('Spin = ' + str(Spin) + '  '+ str(chr(1115)) + '\n'+ '\n'
      +'Single-ion Anisotropy = ' + str(A) + ' (meV) '+'\n'+ '\n'
      +'g factor = '+str(g)+'\n'+ '\n'
      +'Boltzman Constant = ' + str(KB) + ' (meV/Kelvin) '+'\n'+ '\n'
      +'mu_{B} = ' + str(mu_B) + ' (meV/tesla) '+'\n'+ '\n'
      +'Magnetic feild B = ' + str(B) + ' (Tesla) '+'\n'+ '\n'
      +"J\u2081    = {}".format(J[0]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2081    = {}".format(DMI[0]) + ' (meV) '+'\n'+'\n'
      +"J\u2082    = {}".format(J[1]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2082    = {}".format(DMI[1]) + ' (meV) '+'\n'+'\n'
      +"J\u2083    = {}".format(J[2]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2083    = {}".format(DMI[2]) + ' (meV) '+'\n'+'\n'
      +"J\u0302    = {:0.2f}".format(J_tilda) + ' (meV) '+'\n'+'\n'
      +"E(\u0393)  = {:0.3f}".format(E_A_B[0])+ ' (meV) '+'\n'
      +'--------------------------------------------------')
Plot_Band_Structure3D(K_BZ,E_BZ)
#P_DOS(DOS)
if E_A_B[0] < 0.0 :
    print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n')
    print('   #########################################################################')
    print('   #Since the Energy at (\u0393) point is NEGATIVE, the calculation WILL STOP HERE#')
    print('   #########################################################################')
    print('------------------------------------------'+'\n')
    sys.exit()
print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n'
     +'E(Gap) = {:0.3f}'.format(min(E_BZ[1])-max(E_BZ[0]))+' (meV) '+'\n'
     )
print('------------------------------- '
   +'\nPlotting The Berry Phase (2D):'
   +'\n------------------------------- '+'\n') 
Be_phase_im_2D(Be_CR_Beery_plot,K_BZ_Beery_plot)
print('------------------------------- '
   +'\nPlotting The Berry Phase (3D):'
   +'\n------------------------------- '+'\n') 
Be_phase_3D(Be_CR_Beery_plot,K_BZ_Beery_plot)
##############################################
## Calculation Of Thermal Hall Conductivity
##############################################
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##-- Beery_conections_Acoustic
def B_C_A(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_A_star[K0],np.dot(Vy(K0),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vx(K0),psi_A[K0])))
    B_C_2 = ( np.dot(psi_A_star[K0],np.dot(Vx(K0).conj(),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vy(K0).conj(),psi_A[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return 2*B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_A, dataset)
        Ber_CR_A = np.array(result)
Be_CR_A = [round(num, 1) for num in Ber_CR_A]
Chern_Number_A = (sum(Be_CR_A)/(2*np.pi*K_BZ.shape[0]))
print("Chern Number for Acoustic Mode= {:.14f}".format(Chern_Number_A)+'\n'+'\n')
#---------------------
##-- Beery_conections_Optical
def B_C_O(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_O_star[K0],np.dot(Vy(K0),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vx(K0),psi_O[K0])))
    B_C_2 = ( np.dot(psi_O_star[K0],np.dot(Vx(K0).conj(),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vy(K0).conj(),psi_O[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return 2*B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_O, dataset)
        Ber_CR_O = np.array(result)
Be_CR_O = [round(num, 1) for num in Ber_CR_O]
Chern_Number_O = (sum(Be_CR_O)/(2*np.pi*K_BZ.shape[0]))
print("Chern Number for Optical Mode= {:.14f}".format(Chern_Number_O)+'\n'+'\n')
#---------------------
############################################
##-- Calculating The Ocupation Number & C2
############################################
import mpmath as MPM
MPM.mp.dps = 15; MPM.mp.pretty = True
print('---------------------------------------- '
   +'\nCalculating The ocupation number for CrBr\u2083 :'
   +'\n---------------------------------------- '+'\n')
##-- Bos-Einstin Distrbution
def nk_A(t,K): # Acostic Mode
    if t == 0.0 :
        oc_A = 0.0
    else:
        EXP_A = np.exp(np.float128((E_BZ[0,K])/(KB*t)))-1 # Exponential - 1
        oc_A = (1/EXP_A)
    ocup_A = oc_A
    return float("%.6f" %(ocup_A))
##----------------------------
Temp =  [float("%.2f" %(i)) for i in np.arange(intail_Temp,Final_Temp,steps_Temp)]
print('---------------------- '
     +'\n The Acoustic mode :'
     +'\n')
##----- Ocupation number for Acustic mode
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_A,dataset)
n_k_A = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
print('Done.\n')
##----- Ocupation number for Optical mode
###
def nk_O(t,K): # Optical Mode
    if t == 0.0 :
        oc_O = 0
    else:
        EXP_O = np.exp(np.float128((E_BZ[1,K])/(KB*t)))-1 # Exponential - 1
        oc_O = (1/EXP_O)
    ocup_O = oc_O
    return float("%.6f" %(ocup_O))
print('---------------------- '
     +'\n The Optical mode :'
     +'\n')
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_O,dataset)
n_k_O = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
print('Done.\n')
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(n_k_O[i,:],'r*')
#     axis[0].set_title('Ocupation for Optical mode')
#     axis[1].plot(n_k_A[i,:],'bo')
#     axis[1].set_title('Ocupation for Acustic mode')
#     #fig.savefig("Temperature_{}.pdf".format(i))
#     plt.show()
###############----------------------
##--- C2(nk_A)
print('---------------------------------------- '
   +'\nCalculating The C2 for Acoustic mode :'
   +'\n---------------------------------------- '+'\n')
def C2_nk_A(x,y) :
    if n_k_A[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 = ( (1+n_k_A[x,y])*(np.log((1+n_k_A[x,y])/n_k_A[x,y])**2)
               -(np.log(n_k_A[x,y])**2)-(2*MPM.polylog(2, -n_k_A[x,y]))
              )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_A,dataset)
C2_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
print('Done.\n')
##--- C2(nk_O)
print('---------------------------------------- '
   +'\nCalculating The C2 for Optical mode :'
   +'\n---------------------------------------- '+'\n')
def C2_nk_O(x,y) :
    if n_k_O[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 =( (1+n_k_O[x,y])*(np.log((1+n_k_O[x,y])/n_k_O[x,y])**2)
             -(np.log(n_k_O[x,y])**2)-(2*MPM.polylog(2, -n_k_O[x,y]))
             )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_O,dataset)
C2_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
print('Done.\n')
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_O[i,:],'r*')
#     axis[0].set_title('C2 for Optical mode')
#     axis[1].plot(C2_A[i,:],'bo')
#     axis[1].set_title('C2 for Acustic mode')
#     #fig.savefig("Temperature_{}.pdf".format(i))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print('---------------------------------------- '
   +'\nCalculating The C2 * BC Acoustic :'
   +'\n---------------------------------------- '+'\n')
##--- C2 * BC Acustic
def K_xy_A(t,K):
    C2BC_A = C2_A[t,K]*(Be_CR_A[K])
    return C2BC_A
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_A,dataset)
C2_BC_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
print('Done.\n')
print('---------------------------------------- '
   +'\nCalculating The C2 * BC Optical :'
   +'\n---------------------------------------- '+'\n')
##----- C2 * BC Optic
def K_xy_O(t,K):
    C2BC_O = C2_O[t,K]*(Be_CR_O[K])
    return C2BC_O
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_O,dataset)
C2_BC_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
print('Done.\n')
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_BC_O[i,:],'r*')
#     axis[0].set_title('C2 * BC Optical mode')
#     axis[1].plot(C2_BC_A[i,:],'bo')
#     axis[1].set_title('C2 * BC Acustic mode')
#     #fig.savefig("C2BC_Temperature_{}.pdf".format(i))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print('---------------------------------------- '
   +'\nCalculating Thermal Hall Conductivity for CrBr\u2083 :'
   +'\n---------------------------------------- '+'\n')
K_XY_neg = []
K_XY_neg_A = []
K_XY_neg_O = []
for t in range(len(Temp)):
    start_itr_time = datetime.now()
    print('TEMPERATURE = {:.2f}'.format(Temp[t]) + '\n')
    K_XY_O = sum(C2_BC_O[t])/K_BZ.shape[0]
    K_XY_neg_O.append(K_XY_O)
    K_XY_A = sum(C2_BC_A[t])/K_BZ.shape[0]
    K_XY_neg_A.append(K_XY_A)
    cof = (2*np.pi)**-2
    K_XY_total = ((KB**2)*Temp[t])*(K_XY_O + K_XY_A)*cof
    K_XY_neg.append(-K_XY_total) 
    ########################
    end_itr_time = datetime.now()
    print('K_XY_O = '+str(float("%.4f" %((K_XY_neg_O[-1]))))+'\n'
         +'K_XY_A = '+str(float("%.4f" %((K_XY_neg_A[-1]))))+'\n')
    print('Termal Hall Conductivity for CrBr\u2083 = '+str(float("%.4f" %((K_XY_neg[-1]))))+'\n'
         +'\n'+'Duration of each Temperature step: {}'.format(end_itr_time - start_itr_time)
         +'\n---------------------------------')
#---- Ploting THC
def Plot_TH(KXY_p): 
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(KXY_p)) - 1
    end   = int(np.max(KXY_p)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
    #ax.set_xlim(0,Temp[-2])
    ax.plot(Temp,KXY_p,'b', label='DMI = -0.22')
    #ax.plot(Temp,E2,'b', label='DMI = -0.3')
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    ax.set_title("Hall Conductivity ")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa^{xy}x10^{-12}$(W/K)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    #fig.savefig("THC_CrI3.pdf")
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.show()
    return
#---- Ploting THC
def Plot_TH_T(KXY_p,KXY_n): 
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(KXY_p)) - 1
    end   = int(np.max(KXY_p)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
    #ax.set_xlim(0,Temp[-2])
    ax.plot(Temp,KXY_p,'r', label='DMI = 0.22')
    ax.plot(Temp,KXY_n,'b', label='DMI = -0.22')
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    ax.set_title("Hall Conductivity ")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa^{xy}x10^{-12}$(W/K)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    fig.savefig("THC_negative.pdf")
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.show()
    return
# print('--------------------------------\n'
#      +'Ploting THC for Acoustic mode :'
#      +'\n--------------------------------\n')
# Plot_TH(K_XY_neg_A)
# print('Done.\n')
# print('--------------------------------\n'
#      +'Ploting THC for Optical mode :'
#      +'\n--------------------------------\n')
# Plot_TH(K_XY_neg_O)
# print('Done.\n')
print('--------------------------------\n'
     +'Ploting THC :'
     +'\n--------------------------------\n')
Plot_TH_T(K_XY_pos,K_XY_neg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


print('--------\n CrI\u2083 : \n--------\n')
#!/usr/bin/python3.8
from datetime import datetime
start_time = datetime.now()
##-----------
import pickle as pk
import sys
from termcolor import colored
##--
import numpy as np
import math as mt
import scipy.linalg as la
from scipy.linalg import eigh
import scipy.interpolate
from itertools import groupby
import itertools
##----
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = [6.0, 5.0]
plt.rcParams["figure.autolayout"] = True
from matplotlib import cm
#matplotlib.use('Agg') ##-- Solving the plot problem
##---
from multiprocessing import Pool
import multiprocessing as mp
import concurrent.futures
from multiprocessing import cpu_count
processes = cpu_count()
##-------------------
## Useful Functions 
##-------------------
##-- Norm of Vectors
def Norm(vector1):
    return np.linalg.norm(vector1)
# function to convert to superscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal),''.join(super_s))
    return x.translate(res)
##---- display superscipt
##---
##-------------------
##-- Constants
##-------------------
Spin = float("%.1f" %(3/2)) #spin
##-- Single-ion Anisotropy
A = -0.02 #for the SCF
##-- the Lande g factor
g = 2.002
##-- Boltzman Constant : unite is eV/Kelvin
B_C = 0.000086
KB  = float("%.3f" %(B_C*1000)) ##-- in meV/Kelvin
##-- The bohr magneton : unite is eV/tesla
bohr_magneton =0.000057
mu_B =float("%.3f" %(bohr_magneton*1000)) ##-- in meV/tesla
##-- Magnetic feild : unite is Tesla
B = 0.0
##-- Number of Unit cell
N_U = 1
##-------------------------------
##       UNIT CELL              
##-------------------------------
##-- LATTICE_VECTORS
a0 = 6.71  #lattice Constant
a = np.array([ a0,0.000,0.0000])
b = np.array([-a0*0.5,a0*0.86,0.0000])
c = np.array([ 0.000,0.000,12.000])
##-- Atoms Position in Unit Cell
A_P_U_C=np.array(
        [[0.555, 0.777, 0.000] # Atom 1
        ,[0.888, 0.444, 0.000] # Atom 2
        ])
##-- Number of Magnetic Sites in one Unit cell
N_M_S = A_P_U_C.shape[0]
##-------------------------------
##       SUPERCELL             
##-------------------------------
##-- LATTICE_VECTORS
a_m = 3 #a_multiple_by
b_m = 3 #b_multiple_by
c_m = 1 #c_multiple_by
a1  = a*a_m
a2  = b*b_m
a3  = c*c_m
##-- Atoms Position in Supercell
A_P_S = np.zeros([N_M_S,a_m*b_m*c_m,3])
for i in range(A_P_S.shape[0]):
        # Initail Position in Supercell
        A_P_S[i,0] = np.array([
                     float("%.3f" %(A_P_U_C[i,0]/a_m)),
                     float("%.3f" %(A_P_U_C[i,1]/b_m)),
                     float("%.3f" %(A_P_U_C[i,2]/c_m))
                              ])
        for j in range(1,a_m*b_m):
            if j < a_m : # Finding the Position in a1 Direction
                A_P_S[i,j] = A_P_S[i,0]+ np.array([float("%.3f" %(j/a_m)),0.0,0.0]) 
            else : ##-- Finding the Position in a2 Direction
                A_P_S[i,j] = A_P_S[i,j-a_m]+ np.array([0.0,float("%.3f" %(1/b_m)),0.0])
##-------------------------------
##    Neighbors and distance
##-------------------------------
##-- Cartesian Position from Atoms of Supercell
C_P = np.zeros([N_M_S*a_m*b_m,3])
l=0
for j in range(a_m*b_m):
    C_P[l] = A_P_S[0,j,0]*a1+A_P_S[0,j,1]*a2+A_P_S[0,j,2]*a3  #
    C_P[l+1] =A_P_S[1,j,0]*a1+A_P_S[1,j,1]*a2+A_P_S[1,j,2]*a3
    l=l+2 #Counting both atoms
##-- Sites
S1 = int((C_P.shape[0]/2)-1)      #Site 1 from centeral unit cell
S2 = int((C_P.shape[0]/2)-1) + 1  #Site 2 from centeral unit cell
dist=[] #distance between neighbors
for i in range(C_P.shape[0]):
    if i ==  S1:
        continue
    dist.append(float("%.4f" %(Norm(C_P[i]-C_P[S1]))))
dist=sorted(dist)
def nearby_groups(arr, tol_digits=0): #Sorting the list of distance
  # split up sorted input array into groups if they're similar enough
  for (_, grp) in groupby(arr, lambda x: round(x, tol_digits)):
    # yield value from group that is closest to an integer
    yield sorted(grp, key=lambda x: abs(round(x) - x))[0]
array = dist
dist=list(nearby_groups(array))
d1=dist[0]
d2=dist[1]
d3=dist[2]
d4=dist[3]
epsilon = 0.1
##-- Counting the Neighbors
D=0
J1 = 0
J2 = 0
J3 = 0
R_J1 = []
R_J2 = []
R_J3 = []
for i in range(C_P.shape[0]):
    if i != S1:
        D = float("%.4f" %(Norm(C_P[i]-C_P[S1])))
        if abs(D-d1) < epsilon :
            J1=J1+1
            R_J1.append(C_P[i]-C_P[S1])
            #print('r1 = '+ str(C_P[i]-C_P[S1]))
        if abs(D-d2) < epsilon :
            J2=J2+1
            R_J2.append(C_P[i]-C_P[S1])
            #print('r2 = '+ str(C_P[i]-C_P[S1]))
        if abs(D-d3) < epsilon :
            J3=J3+1
            R_J3.append(C_P[i]-C_P[S1])
            #print('r3 = '+ str(C_P[i]-C_P[S1])+'\n-------------')
del R_J2[3:]
# print('-----------Neighbors vectors FM Configuration---------------\n')
# print('The First  Neighbor = ' + str(R_J1) + '\n'
#      +'The Second Neighbor = ' + str(R_J2) + '\n'
#      +'The Third  Neighbor = ' + str(R_J3))
# print('---------------------------------------------------------------------\n')
print('-----------------------------------------------------------\n'
      +'The Nearest Neighbors for each site in FM Configuration :\n'
      +'The 1{} Neighbor = '.format(get_super('st')) + str(J1) + '\n'
      +'The 2{} Neighbor = '.format(get_super('nd')) + str(J2) + '\n'
      +'The 3{} Neighbor = '.format(get_super('rd')) + str(J3) + '\n'
      +'--------------------------------------------\n')
##------------------------
## K space griding (Start)
##------------------------
print('\n'+'Calculating the 1BZ : '+'\n')
##-- Reciprocal Lattice Vectors Unit Cell
b1 = (2*np.pi*(np.cross(b,c)))/np.dot(a,np.cross(b,c))
b2 = (2*np.pi*(np.cross(c,a)))/np.dot(a,np.cross(b,c))
b3 = (2*np.pi*(np.cross(a,b)))/np.dot(a,np.cross(b,c))
print('----------- Unit Cell Vectors in Reciprocal Space ------------'+'\n')
print( "b\u2081 = {}".format(b1)+'\n'
      +"b\u2082 = {}".format(b2)+'\n'
      +"b\u2083 = {}".format(b3)+'\n'
      +'---------------------------------------------'+'\n')
##-----------
##-- The heigh path is Gamma(0,0,0)--->K(0,0.5,0)--->M(0.5,0,0)--->Gamma(0,0,0)
G  = (0.0*b1)      + (0.0*b2)      + (0.0*b3) ##--- Gamma
K1 = ((1./3)*b1)   + ((1./3)*b2)   + (0.0*b3)
K2 = ((-1./3)*b1)  + ((2./3)*b2)   + (0.0*b3)
K3 = (((-2./3)*b1) + ((1./3)*b2)   + (0.0*b3))
K4 = ((-1./3)*b1)  + ((-1./3)*b2)  + (0.0*b3) ##--- (-K1)
K5 = ((1./3)*b1)   + ((-2./3)*b2)  + (0.0*b3) ##--- (-K2)
K6 = ((2./3)*b1)   + ((-1./3)*b2)  + (0.0*b3) ##--- (-K3)
M  = (0.5*b1)      + (0.0*b2)      + (0.0*b3) ##--- M
##--    2D plot of 1BZ
sd_Kx = int(input("Enter your K\u2081 subdivisions : ")) #subdivisions Kx
sd_Ky = int(input("Enter your K\u2082 subdivisions : ")) #subdivisions Ky
sd_Kz = int(input("Enter your K\u2083 subdivisions : ")) #subdivisions Kz
print('\n---------------------------------------------')
##--------------- Ploting the 1st unite cell in reciprocal space
def Plot_1st_U_R_S(K_Space):
    plt.plot(K_Space[:,0],K_Space[:,1],'k.')
    ##-- Gamma Point
    plt.text(G[0],G[1]+.02,'\u0393',color='red' , fontsize=20)
    plt.plot(G[0],G[1],'ro')
    ##-- K1 Point
    plt.text(K1[0]-.08, K1[1]-.08,"$K1$",color='red', fontsize=20)
    plt.plot(K1[0],K1[1],'ro')
    ##-- K2 prime Point
    plt.text(K2[0]+.01, K2[1]-.06,"$K2$",color='red', fontsize=20)
    plt.plot(K2[0],K2[1],'ro')
    ##-- K3 Point
    plt.text(K3[0]+.03, K3[1],"$K3$",color='red', fontsize=20)
    plt.plot(K3[0],K3[1],'ro')
    ##-- K4 Point
    plt.text(K4[0]+.01, K4[1]+.01,"$K4$",color='red', fontsize=20)
    plt.plot(K4[0],K4[1],'ro')
    ##-- K5 Point
    plt.text(K5[0]-.08, K5[1]+.03,"$K5$",color='red', fontsize=20)
    plt.plot(K5[0],K5[1],'ro')
    ##-- K6 Point
    plt.text(K6[0]-.11, K6[1]-.01,"$K6$",color='red', fontsize=20)
    plt.plot(K6[0],K6[1],'ro')
    ##--
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")
    plt.title('K points in the 1BZ')
#     plt.savefig("1_Unit_cell_in_RS.pdf")
#     plt.close()
    plt.show()
    return
##--------------- Ploting the 1st unite cell in reciprocal space
def Plot_1BZ_3D(K_Space):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ##-----------------
    x = K_Space[:,0]
    y = K_Space[:,1]
    ##-----------------
    ax.scatter(x, y, c='m', marker='*')
    ax.view_init(elev=90., azim=-90)
    ax.set_xticks([round(min(x),1),0.0,round(max(x),1)])
    ax.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ax.set_zticks([])
    #-----------
    ax.set_xlabel("$K_{x}$")
    ax.set_ylabel("$K_{y}$")
    ##-- Gamma Point
    ax.text(G[0]+.02,G[1]+.02,G[2]+.02,'\u0393',color='black' , fontsize=10)
    ax.scatter(G[0], G[1], c='k', marker='o')
    ##-- K1 Point
    ax.text(K1[0]-.08, K1[1]-.08, K1[2],"$K_1$",color='black', fontsize=10)
    ax.scatter(K1[0], K1[1], c='k', marker='o')
    ##-- K2 prime Point
    ax.text(K2[0]+.01, K2[1]-.06, K2[2]+.01,"$K_2$",color='black', fontsize=10)
    ax.scatter(K2[0], K2[1], c='k', marker='o')
    ##-- K3 Point
    ax.text(K3[0]+.03, K3[1],K3[2],"$K_3$",color='black', fontsize=10)
    ax.scatter(K3[0], K3[1], c='k', marker='o')
    ##-- K4 Point
    ax.text(K4[0]+.01, K4[1]+.01,K4[2]+.01,"$K_4$",color='black', fontsize=10)
    ax.scatter(K4[0], K4[1], c='k', marker='o')
    ##-- K5 Point
    ax.text(K5[0]-.08, K5[1]+.03,K5[2]+.01,"$K_5$",color='black', fontsize=10)
    ax.scatter(K5[0], K5[1], c='k', marker='o')
    ##-- K6 Point
    ax.text(K6[0]-.11, K6[1]-.01,K6[2]+.01,"$K_6$",color='black', fontsize=10)
    ax.scatter(K6[0], K6[1], c='k', marker='o')
    ##-- M Point
    ax.text(M[0]+.01, M[1]+.01,M[2]+.01,"$M$",color='green', fontsize=10)
    ax.scatter(M[0], M[1], c='g', marker='o')
    ##--
    ax.plot([G[0],K1[0]], [G[1], K1[1]],[G[2], K1[2]])
    ax.plot([G[0],M[0]],  [G[1], M[1]], [G[2], M[2]])
    ##--
    ax.set_title('K points in the 1BZ')
#     plt.savefig("1st_BZ_3D.pdf")
#     plt.close()
    plt.show()
    return
def Plot_1BZ():
    x1 = []
    y1 = []
    for i in range(0,sd_Kx,1):
        for j in range(0,sd_Ky,1):
            x1.append(K_BZ1[(i*sd_Kx)+j,0])
            y1.append(K_BZ1[(i*sd_Kx)+j,1])
    ##-- Gamma Point
    plt.text(G[0]-.02,G[1]+.02,'\u0393',color='blue' , fontsize=20)
    plt.plot(G[0],G[1],'ro')
    ##-- K1 Point
    plt.text(K1[0]-.08, K1[1]-.08,"$K1$",color='blue', fontsize=20)
    plt.plot(K1[0],K1[1],'ro')
    ##-- K2 Point
    plt.text(K2[0]+.01, K2[1]-.06,"$K2$",color='blue', fontsize=20)
    plt.plot(K2[0],K2[1],'ro')
    ##-- K3 Point
    plt.text(K3[0]+.03, K3[1],"$K3$",color='blue', fontsize=20)
    plt.plot(K3[0],K3[1],'ro')
    ##-- K4 Point
    plt.text(K4[0]+.01, K4[1]+.01,"$K4$",color='blue', fontsize=20)
    plt.plot(K4[0],K4[1],'ro')
    ##-- K5 Point
    plt.text(K5[0]-.08, K5[1]+.03,"$K5$",color='blue', fontsize=20)
    plt.plot(K5[0],K5[1],'ro')
    ##-- K6 Point
    plt.text(K6[0]-.11, K6[1]-.01,"$K6$",color='blue', fontsize=20)
    plt.plot(K6[0],K6[1],'ro')
    ##-- K6 Point
    plt.text(M[0]+.01, M[1]+.01,"$M$",color='green', fontsize=20)
    plt.plot(M[0],M[1],'go')
    ##-- lable of axis
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")
    ##-- Arrow plots
    plt.arrow(K1[0],K1[1],(K2-K1)[0],(K2-K1)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K2[0],K2[1],(K3-K2)[0],(K3-K2)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K3[0],K3[1],(K4-K3)[0],(K4-K3)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K4[0],K4[1],(K5-K4)[0],(K5-K4)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K5[0],K5[1],(K6-K5)[0],(K6-K5)[1],head_width = 0.0001, width = 0.002,ec ='black')
    plt.arrow(K6[0],K6[1],(K1-K6)[0],(K1-K6)[1],head_width = 0.0001, width = 0.002,ec ='black')
    ##
    plt.arrow(G[0], G[1], K1[0], K1[1],head_width = 0.0001, width = 0.002,ec ='red')
    plt.arrow(G[0], G[1], M[0], M[1], head_width = 0.0001, width = 0.002,ec ='green')
    plt.grid()
    plt.title('1BZ and High Symmetry Path')
    ##-- savinf plot
#     plt.savefig("1BZ.pdf")
#     plt.close()
    plt.show()
    return
###-------------------------
print('\n---------------------\n Sampling the 1BZ : \n---------------------')
##-- b1 > 0 & b2 > 0
def K_mesh1(i,j):
    K_BZ = (i/sd_Kx)*(b1)+(j/sd_Ky)*(b2)
    return K_BZ
if __name__ == '__main__':
    # Define the dataset
    dataset = [(i,j) for i in np.arange(0,sd_Kx+1,1) for j in np.arange(0,sd_Ky+1,1)]
    cpus = mp.cpu_count()
    with Pool(processes=cpus) as pool:
        res1 = pool.starmap(K_mesh1, dataset)
K_BZ1 =np.array(res1)
##-- b1 < 0 & b2 > 0
def K_mesh2(i,j):
    K_BZ = (i/sd_Kx)*(-b1)+(j/sd_Ky)*(b2)
    return K_BZ
if __name__ == '__main__':
    # Define the dataset
    dataset = [(i,j) for i in np.arange(1,sd_Kx+1,1) for j in np.arange(0,sd_Ky+1,1)]
    cpus = mp.cpu_count()
    with Pool(processes=cpus) as pool:
        res2 = pool.starmap(K_mesh2, dataset)
K_BZ2 =np.array(res2)
##-- b1 > 0 & b2 < 0
def K_mesh3(i,j):
    K_BZ = (i/sd_Kx)*(b1)+(j/sd_Ky)*(-b2)
    return K_BZ
if __name__ == '__main__':
    # Define the dataset
    dataset = [(i,j) for i in np.arange(0,sd_Kx+1,1) for j in np.arange(1,sd_Ky+1,1)]
    cpus = mp.cpu_count()
    with Pool(processes=cpus) as pool:
        res3 = pool.starmap(K_mesh3, dataset)
K_BZ3 = np.array(res3)
##-- b1 < 0 & b2 < 0
def K_mesh4(i,j):
    K_BZ = (i/sd_Kx)*(-b1)+(j/sd_Ky)*(-b2)
    return K_BZ
if __name__ == '__main__':
    # Define the dataset
    dataset = [(i,j) for i in np.arange(1,sd_Kx+1,1) for j in np.arange(1,sd_Ky+1,1)]
    cpus = mp.cpu_count()
    with Pool(processes=cpus) as pool:
        res4 = pool.starmap(K_mesh4, dataset)
K_BZ4 =np.array(res4)
#-- finding the K points in the 1BZ
K_BZ = np.zeros([(K_BZ1.shape[0]+K_BZ2.shape[0]+K_BZ3.shape[0]+K_BZ4.shape[0]),3])
K_BZ[0:K_BZ1.shape[0]] = K_BZ1
K_BZ[K_BZ1.shape[0]:K_BZ1.shape[0]+K_BZ2.shape[0]] = K_BZ2
K_BZ[K_BZ1.shape[0]+K_BZ2.shape[0]:K_BZ1.shape[0]+K_BZ2.shape[0]+K_BZ3.shape[0]] = K_BZ3
K_BZ[K_BZ1.shape[0]+K_BZ2.shape[0]+K_BZ3.shape[0]:] = K_BZ4
K_BZ_intial = K_BZ
K_BZ_Beery_plot = K_BZ
#---------------
K_BZ_x_limit = []
for i in K_BZ:
    if (i[0] <= round(K6[0],3) and i[0] >= 0) :
        K_BZ_x_limit.append(i)
    if (i[0] >= round(K3[0],3) and i[0] < 0):
        K_BZ_x_limit.append(i)
K_BZ_y_limit = []
for i in K_BZ_x_limit:
    if (i[1] <= round(K2[1],3) and i[1] > 0 ) :
        K_BZ_y_limit.append(i)
    if (i[1] >= round(K5[1],3) and i[1] < 0):
        K_BZ_y_limit.append(i)
K_BZ = np.zeros([len(K_BZ_y_limit),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_y_limit[i]
# ###-------------------------------------------------
# ##-- Definig the Equation of lines for the path
# ###--------------------------------------------
#--K3_K2
K_BZ_K3_K2 = []
def K3_K2(vec): 
    if K3[0]==K2[0] :
        K3K2 = abs(vec[0]-K3[0])
    elif K3[1]==K2[1] :
        K3K2 = abs(vec[1]-K3[1])
    else :
        K3K2 = vec[1]-(((K3[1]-K2[1])/(K3[0]-K2[0]))*(vec[0]-K2[0])+K2[1])
    return K3K2
for i in K_BZ_y_limit:
    if float("%.4f" %(K3_K2(i))) < 0:
        K_BZ_K3_K2.append(i)
    if float("%.4f" %(K3_K2(i))) == 0:
        K_BZ_K3_K2.append(i)
K_BZ = np.zeros([len(K_BZ_K3_K2),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_K3_K2[i]
#--K4_K3
K_BZ_K4_K3 = []
def K4_K3(vec): 
    if K4[0]==K3[0] :
        K4K3 = abs(vec[0]-K4[0])
    elif K4[1]==K3[1] :
        K4K3 = abs(vec[1]-K4[1])
    else :
        K4K3 = vec[1]-(((K4[1]-K3[1])/(K4[0]-K3[0]))*(vec[0]-K3[0])+K3[1])
    return K4K3
##--
for i in K_BZ_K3_K2:
    if float("%.4f" %(K4_K3(i))) > 0:
        K_BZ_K4_K3.append(i)
    if float("%.4f" %(K4_K3(i))) == 0:
        K_BZ_K4_K3.append(i)
K_BZ = np.zeros([len(K_BZ_K4_K3),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_K4_K3[i]
#--K6_K5
K_BZ_K6_K5 = []
def K6_K5(vec): 
    if K6[0]==K5[0] :
        K6K5 = abs(vec[0]-K6[0])
    elif K6[1]==K5[1] :
        K6K5 = abs(vec[1]-K6[1])
    else :
        K6K5 = vec[1]-(((K6[1]-K5[1])/(K6[0]-K5[0]))*(vec[0]-K5[0])+K5[1])
    return K6K5
##--
for i in K_BZ_K4_K3:
    if float("%.4f" %(K6_K5(i))) > 0:
        K_BZ_K6_K5.append(i)
    if float("%.4f" %(K6_K5(i))) == 0:
        K_BZ_K6_K5.append(i)
K_BZ = np.zeros([len(K_BZ_K6_K5),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_K6_K5[i]
#--K6_K5
K_BZ_K1_K6 = []
def K1_K6(vec): 
    if K1[0]==K6[0] :
        K1K6 = abs(vec[0]-K1[0])
    elif K1[1]==K6[1] :
        K1K6 = abs(vec[1]-K1[1])
    else :
        K1K6 = vec[1]-(((K1[1]-K6[1])/(K1[0]-K6[0]))*(vec[0]-K6[0])+K6[1])
    return K1K6
##--
for i in K_BZ_K6_K5:
    if float("%.4f" %(K1_K6(i))) < 0:
        K_BZ_K1_K6.append(i)
    if float("%.4f" %(K1_K6(i))) == 0:
        K_BZ_K1_K6.append(i)
K_BZ = np.zeros([len(K_BZ_K1_K6),3])
for i in range(K_BZ.shape[0]) :
    K_BZ[i] = K_BZ_K1_K6[i]
##--------
Plot_1st_U_R_S(K_BZ_intial)
Plot_1st_U_R_S(K_BZ)
Plot_1BZ_3D(K_BZ)
##------------------
## End of BZ
##------------------
Plot_1BZ()      ##-- ploting the 1BZ
print('1BZ is calculated.\n')
##------------------------
## K space griding (End)
##------------------------

##---------------------
##-- Exchange Interaction parameter for the first, second,third neighbors [J1,J2,J3] 
##---------------------
J=[-1.48,-0.08,0.11]
##-- The sum of all J’s on one specific spin site
J_tilda = (J1*J[0])+(J2*J[1])+(J3*J[2])
E_g = N_M_S*N_U*((-J_tilda*(Spin**2))-(Spin*g*mu_B*B)-(A*(Spin**2)))##-- Ground state energy
##-- DMI coeficient
DMI = [0.0,0.0,0.0]
##-- phase
if J[1] == 0 and DMI[1]== 0 :
    phase = 0.0
elif J[1] == 0 :
    phase =np.pi/2 # if J2 = 0.0
elif J[1] != 0 :
    phase = np.arctan(DMI[1]/(J[1]))
print('\n---------------------------------------------\n'
      +"Calculation of Magnon Bands for DMI =  {:.3f}".format(DMI[1])
      +'\n--------------------------------------------\n')
##-------------------------------
##   Function for calculating the J_K 
##-------------------------------
##-- The J_K between the Site A and B
def J_K_AB(K) :
    AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        AB_FT.append(np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        AB_FT.append(np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    J_AB = np.sum(AB_FT)
    return J_AB
##-- The J_K between the Site A and A
def J_K_AA(K) :
    AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        AA_FT.append(mt.cos(phase+np.dot(K,R_J2[p])))
    J_AA = np.sum(AA_FT)
    return J_AA
##-- The J_K between the Site B and B
def J_K_BB(K) :
    BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        BB_FT.append(mt.cos(phase-np.dot(K,R_J2[p])))
    J_BB = np.sum(BB_FT)
    return J_BB
##---------------------------------------------------------------------------
##   Function for calculating the Hamiltonian and oocupation number(Strat)
##---------------------------------------------------------------------------
##-- Function for the Hamiltonians 
def H_HSP(K0):
    Ha_L = np.zeros([2,2],dtype=complex)
    Ha_L[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_AA(K_H_P[K0]))-(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B
    Ha_L[0,1] = 2*Spin*J_K_AB(K_H_P[K0])
    Ha_L[1,0] = np.conjugate(Ha_L[0,1])
    Ha_L[1,1] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_BB(K_H_P[K0]))-(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B
    H_L = Ha_L
    return H_L   #-- Hamiltonian of LSWT
###--- EigenValues function for K points
E_A_HSP = []
psi_A_HSP = []
psi_A_star_HSP = []
E_O_HSP = []
psi_O_HSP = []
psi_O_star_HSP = []
###--- Wave function for K points
def PSI_A_O_HSP(kpoint):
    v_L_B, w_L_B = eigh(H_HSP(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(PSI_A_O, dataset)
        PSI_L = result
for i in range(K_BZ.shape[0]):
    E_A_B.append(PSI_L[i][0][0])
    psi_A.append(PSI_L[i][1][0])
    psi_A_star.append(PSI_L[i][1][0].reshape(1,2).conj())
    ########
    E_O_B.append(PSI_L[i][0][1])
    psi_O.append(PSI_L[i][1][1])
    psi_O_star.append(PSI_L[i][1][1].reshape(1,2).conj())
def Plot_Band_Structure1(E1,E2): # E1 Energy for Acoustic mode & E2 for Optical mode
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(E1)) - 1
    end   = int(np.max(E2)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax.plot(plot_path_B,E1,'r', label='A')
    ax.plot(plot_path_B,E2,'b', label='O')
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    label=( '\u0393','$K$',"$M$",'\u0393')
    ##-- put title
    ax.set_title("Band Energy of High Symmetry Path")
    ax.set_xlabel0('K path')
    ax.set_ylabel("Energy (meV)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig.savefig('Band_Energy_of_High_Symmetry_Path.pdf')
    plt.show()
    return
##----- Hamiltionian
def B_matrix(K0):
    Ha_B = np.zeros([2,2],dtype=complex) 
    ##############
    Ha_B[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_AA(K_BZ[K0]))-(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B
    ##############
    Ha_B[0,1] = 2*Spin*J_K_AB(K_BZ[K0])
    ##############
    Ha_B[1,0] = np.conjugate(Ha_B[0,1])
    ##############   
    Ha_B[1,1] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_BB(K_BZ[K0]))-(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B
    H_B = Ha_B
    return H_B #-- Hamiltonian of LSWT
##------------------------------------------------------------------------
##   Function for calculating the Hamiltonian and oocupation number (End)
##------------------------------------------------------------------------
##------------------------------------------------
##  Calculating the magnon-band for LSWT (Start)
##------------------------------------------------
##--
###--- EigenValues function for K points
E_A_B = []
psi_A = []
psi_A_star = []
E_O_B = []
psi_O = []
psi_O_star = []
###--- Wave function for K points
def PSI_A_O(kpoint):
    v_L_B, w_L_B = eigh(B_matrix(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(PSI_A_O, dataset)
        PSI_L = result
for i in range(K_BZ.shape[0]):
    E_A_B.append(PSI_L[i][0][0] )
    psi_A.append(PSI_L[i][1][0])
    psi_A_star.append(PSI_L[i][1][0].reshape(1,2).conj())
    ########
    E_O_B.append(PSI_L[i][0][1])
    psi_O.append(PSI_L[i][1][1])
    psi_O_star.append(PSI_L[i][1][1].reshape(1,2).conj())
E_BZ = np.zeros([N_M_S,K_BZ.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ[0,i] = E_A_B[i]
    E_BZ[1,i] = E_O_B[i]
en_cross = round(min(E_BZ[1,:]))
def Plot_Band_Structure3D(K,E): # E1 Energy for Acoustic mode & E2 for Optical mode
    figure = plt.figure(figsize=(6,5))
    ##--
    axis = figure.add_subplot(111, projection = '3d')
    x = K[:,0]
    y = K[:,1]
    en_ac = np.array([E[0,:],E[0,:]])
    en_op = np.array([E[1,:],E[1,:]])
    ##--
    axis.plot_wireframe(x, y, en_op, cstride=1, color='r', label='Optical')
    axis.plot_wireframe(x, y, en_ac, rstride=1, color='b', label='Acostic')
    axis.legend(loc='best', bbox_to_anchor=(0.35,0.77))
    axis.view_init(elev=4., azim=45)
    ##--
    axis.set_xlabel("$K_{x}$")
    axis.set_ylabel("$K_{y}$")
    axis.zaxis.set_rotate_label(False)  # disable automatic rotation
    axis.set_zlabel("Energy (meV)", rotation=90)
    axis.text2D(0.3, 0.79, "Magnon band structure", transform=axis.transAxes)
    ##-------------
    axis.set_xticks([round(min(x),1)+.1,0.0,round(max(x),1)-.1])
    axis.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [round(min(E[1,:])),round(min(E[1,:])),round(min(E[1,:])),
         round(min(E[1,:])),round(min(E[1,:])),round(min(E[1,:])),
         round(min(E[1,:]))]
             ])
    axis.scatter(x1, y1,z1 , color='c')
    plt.rcParams.update({'figure.max_open_warning': 0})
    figure.tight_layout()
    # make an PDF figure of a plot'Band_Structure_DMI_{}.pdf'.format(DMI[1])
    figure.savefig('Band_Structure_DMI_{}.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
##------------------------------------------
##   Function for Magnon-DOS Plots (Start)
##------------------------------------------
DOS_magnon = E_A_B
for i in E_O_B:
    DOS_magnon.append(i)
##-- now plot density of states
##-- now plot density of states
def P_DOS(E3):
    fig, ax = plt.subplots()
    ax.hist(E3,80,range=(np.min(E3),np.max(E3)))
    ax.set_ylim(0.0,100.0)
    # put title
    ax.set_title("Density of states")
    ax.set_xlabel("Band energy")
    ax.set_ylabel("Number of states")
    # make an PDF figure of a plot
    fig.tight_layout()
    fig.savefig("Inversion-symmetry_dos_CrBr3.pdf")
    plt.show()
    return
print('Spin = ' + str(Spin) + '  '+ str(chr(1115)) + '\n'+ '\n'
      +'Single-ion Anisotropy = ' + str(A) + ' (meV) '+'\n'+ '\n'
      +'g factor = '+str(g)+'\n'+ '\n'
      +'Boltzman Constant = ' + str(KB) + ' (meV/Kelvin) '+'\n'+ '\n'
      +'mu_{B} = ' + str(mu_B) + ' (meV/tesla) '+'\n'+ '\n'
      +'Magnetic feild B = ' + str(B) + ' (Tesla) '+'\n'+ '\n'
      +"J\u2081    = {}".format(J[0]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2081    = {}".format(DMI[0]) + ' (meV) '+'\n'+'\n'
      +"J\u2082    = {}".format(J[1]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2082    = {}".format(DMI[1]) + ' (meV) '+'\n'+'\n'
      +"J\u2083    = {}".format(J[2]) + ' (meV) '+'\n'+'\n'
      +"DMI\u2083    = {}".format(DMI[2]) + ' (meV) '+'\n'+'\n'
      +"J\u0302    = {:0.2f}".format(J_tilda) + ' (meV) '+'\n'+'\n'
      +"E(\u0393)  = {:0.3f}".format(E_A_B[0])+ ' (meV) '+'\n'
      +'--------------------------------------------------')
Plot_Band_Structure3D(K_BZ,E_BZ)
#P_DOS(DOS_magnon)
##---------------------------
##-- Considering DMI! = 0.0
##--------------------------
DMI = [0.0,0.22,0.0]
##-- phase
if J[1] == 0 and DMI[1]== 0 :
    phase = 0.0
elif J[1] == 0 :
    phase =np.pi/2 # if J2 = 0.0
elif J[1] != 0 :
    phase = np.arctan(DMI[1]/(J[1]))
##--------------------------------
##  The Velocity Functions (Start)
##--------------------------------
#-- VX
##-- The velocity Site A and B
def VX_AB(K) :
    x_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        x_AB_FT.append(R_J1[q][0]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        x_AB_FT.append(R_J3[q][0]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_x_AB = np.sum(x_AB_FT)
    return V_x_AB
##-- The velocity Site A and A
def VX_AA(K) :
    x_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_AA_FT.append(R_J2[p][0]*mt.sin(phase-np.dot(K,R_J2[p])))
    V_x_AA = np.sum(x_AA_FT)
    return V_x_AA
##-- The velocity Site B and B
def VX_BB(K) :
    x_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_BB_FT.append(R_J2[p][0]*mt.sin(phase+np.dot(K,R_J2[p])))
    V_x_BB = np.sum(x_BB_FT)
    return V_x_BB
#-- Vy
##-- The velocity Site A and B
def VY_AB(K) :
    y_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        y_AB_FT.append(R_J1[q][1]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        y_AB_FT.append(R_J3[q][1]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_y_AB = np.sum(y_AB_FT)
    return V_y_AB
##-- The velocity Site A and A
def VY_AA(K) :
    y_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_AA_FT.append(R_J2[p][1]*mt.sin(phase-np.dot(K,R_J2[p])))
    V_y_AA = np.sum(y_AA_FT)
    return V_y_AA
##-- The velocity Site B and B
def VY_BB(K) :
    y_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_BB_FT.append(R_J2[p][1]*mt.sin(phase+np.dot(K,R_J2[p])))
    V_y_BB = np.sum(y_BB_FT)
    return V_y_BB
##--------------------------------
##  The Velocity Functions (End)
##--------------------------------
#-------------------------------
## Ploting Beery Phase Colorbar
##------------------------------
#---- Sampling 1BZ
K_BZ_Beery_plot_x_limit = []
for i in K_BZ_Beery_plot:
    if (i[0] <= round(K6[0],3)+.1 and i[0] >= 0) :
        K_BZ_Beery_plot_x_limit.append(i)
    if (i[0] >= round(K3[0],3)-.1 and i[0] < 0):
        K_BZ_Beery_plot_x_limit.append(i)
K_BZ_Beery_plot_y_limit = []
for i in K_BZ_Beery_plot_x_limit:
    if (i[1] <= round(K2[1],3)+.2 and i[1] > 0 ) :
        K_BZ_Beery_plot_y_limit.append(i)
    if (i[1] >= round(K5[1],3)-.2 and i[1] < 0):
        K_BZ_Beery_plot_y_limit.append(i)
K_BZ_Beery_plot = np.zeros([len(K_BZ_Beery_plot_y_limit),3])
for i in range(K_BZ_Beery_plot.shape[0]) :
    K_BZ_Beery_plot[i] = K_BZ_Beery_plot_y_limit[i]
##----- Hamiltionian
def B_matrix_Beery_plot(K0):
    Ha_B_Beery_plot = np.zeros([2,2],dtype=complex) 
    ##############
    Ha_B_Beery_plot[0,0] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_AA(K_BZ_Beery_plot[K0]))
                -(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B )
    ##############
    Ha_B_Beery_plot[0,1] = 2*Spin*J_K_AB(K_BZ_Beery_plot[K0])
    ##############
    Ha_B_Beery_plot[1,0] = np.conjugate(Ha_B_Beery_plot[0,1])
    ##############   
    Ha_B_Beery_plot[1,1] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_BB(K_BZ_Beery_plot[K0]))-(2*Spin*J_tilda)
                -(2*Spin*A)+(g*mu_B)*B)
    H_B_Beery_plot = Ha_B_Beery_plot
    return H_B_Beery_plot #-- Hamiltonian of LSWT
###--- EigenValues function for K points
E_A_B_Beery_plot = []
psi_A_Beery_plot = []
psi_A_star_Beery_plot = []
E_O_B_Beery_plot = []
psi_O_Beery_plot = []
psi_O_star_Beery_plot = []
###--- Wave function for K points
def PSI_A_O_Beery_plot(kpoint):
    v_L_B, w_L_B = eigh(B_matrix_Beery_plot(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result_Beery_plot = pool.map(PSI_A_O_Beery_plot, dataset)
        PSI_L_Beery_plot = result_Beery_plot
for i in range(K_BZ_Beery_plot.shape[0]):
    E_A_B_Beery_plot.append(PSI_L_Beery_plot[i][0][0])
    psi_A_Beery_plot.append(PSI_L_Beery_plot[i][1][0])
    psi_A_star_Beery_plot.append(PSI_L_Beery_plot[i][1][0].reshape(1,2).conj())
    ########
    E_O_B_Beery_plot.append(PSI_L_Beery_plot[i][0][1])
    psi_O_Beery_plot.append(PSI_L_Beery_plot[i][1][1])
    psi_O_star_Beery_plot.append(PSI_L_Beery_plot[i][1][1].reshape(1,2).conj())
E_BZ_Beery_plot = np.zeros([N_M_S,K_BZ_Beery_plot.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ_Beery_plot[0,i] = E_A_B_Beery_plot[i]
    E_BZ_Beery_plot[1,i] = E_O_B_Beery_plot[i]
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ_Beery_plot[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ_Beery_plot[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##----- Calculating the Chern Numbers
##-- Berry Function
def Beery_conections_plot(K0):
    beery_cof = 1./((E_O_B_Beery_plot[K0]-E_A_B_Beery_plot[K0])**2)
    B_C_1_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vy(K0),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vx(K0),psi_O_Beery_plot[K0])))
    B_C_2_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vx(K0).conj(),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vy(K0).conj(),psi_O_Beery_plot[K0])))
    B_C_3_plot = beery_cof*(B_C_1_plot+B_C_2_plot)
    return B_C_3_plot[0].imag
Be_CR = 0
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(Beery_conections_plot, dataset)
        Ber_CR_Beery_plot = np.array(result)
Be_CR_Beery_plot = [round(num, 1) for num in Ber_CR_Beery_plot]
def Be_phase_im_2D(Beery,K):
    # convert to arrays to make use of previous answer to similar question
    x = K[:,0]
    y = K[:,1]
    z = np.asarray(Beery)

    # Set up a regular grid of interpolation points
    nInterp = 400
    xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(), y.max(), nInterp)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate; there's also method='cubic' for 2-D data such as here
    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

    img = plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower', interpolation='nearest',
                     cmap='coolwarm',extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto') 
    plt.xlim(x.min()-.02,x.max()+.02)
    plt.ylim(y.min()-.03,y.max()+.03)
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")        
    cbar = plt.colorbar(img)
    cbar.ax.get_yaxis().labelpad = 7
    cbar.ax.set_ylabel('Beery cervature', rotation=90)
    plt.savefig('Berry_phase_DMII_{}_2D.pdf'.format(DMI[1]),dpi=1200,bbox_inches="tight")
    plt.show()
    return
#---------------------
def Be_phase_3D(Berry,K):
    X = K[:,0]
    Y = K[:,1]
    Z = np.array([(Berry),(Berry)])
    # Normalize to [0,1]
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.coolwarm(norm(Z))
    rcount, ccount, _ = colors.shape
    zlim = (0,0.1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    #--------------------------------
    ax.view_init(elev=25., azim=45)
    ax.set_xlabel("$K_{x}$")
    ax.set_ylabel("$K_{y}$")
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("Berry phase", rotation=90)
    ##-------------
    ax.set_xticks([round(min(X),1)+0.2,0.0,round(max(X),1)-0.2])
    ax.set_yticks([round(min(Y),1)+0.2,0.0,round(max(Y),1)-0.2])
    ##--------------
    fig.savefig('Berry_phase_DMII_{}_3D.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
#------------------------------------------------------------------
##-- Calculating Beery cervature & Chern number
print('---------------------------------------------------------------- '
   +'\nCalculating Beery cervature & Chern number (General Formulas)  :'
   +'\n---------------------------------------------------------------- '+'\n') 
###--- EigenValues function for K points
E_A_B = []
psi_A = []
psi_A_star = []
E_O_B = []
psi_O = []
psi_O_star = []
###--- Wave function for K points
def PSI_A_O(kpoint):
    v_L_B, w_L_B = eigh(B_matrix(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(PSI_A_O, dataset)
        PSI_L = result
for i in range(K_BZ.shape[0]):
    E_A_B.append(PSI_L[i][0][0])
    psi_A.append(PSI_L[i][1][0])
    psi_A_star.append(PSI_L[i][1][0].reshape(1,2).conj())
    ########
    E_O_B.append(PSI_L[i][0][1])
    psi_O.append(PSI_L[i][1][1])
    psi_O_star.append(PSI_L[i][1][1].reshape(1,2).conj())
E_BZ = np.zeros([N_M_S,K_BZ.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ[0,i] = E_A_B[i]
    E_BZ[1,i] = E_O_B[i]
##------------------------------------------
##   Function for Magnon-DOS Plots (Start)
##------------------------------------------
DOS = E_A_B
for i in E_O_B:
    DOS.append(i)
##-- now plot density of states
def P_DOS(E3):
    fig, ax = plt.subplots()
    ax.hist(E3,80,range=(np.min(E3),np.max(E3)))
    ax.set_ylim(0.0,100.0)
    # put title
    ax.set_title("Density of states")
    ax.set_xlabel("Band energy")
    ax.set_ylabel("Number of states")
    # make an PDF figure of a plot
    fig.tight_layout()
    #fig.savefig("Inversion-symmetry_dos_CrBr3.pdf")
    plt.show()
def Plot_Band_Structure3D(K,E): # E1 Energy for Acoustic mode & E2 for Optical mode
    figure = plt.figure(figsize=(6,5))
    axis = figure.add_subplot(111, projection = '3d')
    ##-----------------
    x = K[:,0]
    y = K[:,1]
    en_ac = np.array([E[0,:],E[0,:]])
    en_op = np.array([E[1,:],E[1,:]])
    ##----------------
    # axis.scatter(x, y, c='k', marker='*')
    axis.plot_wireframe(x, y, en_op, cstride=1, color='r', label='Optical')
    axis.plot_wireframe(x, y, en_ac, rstride=1, color='b', label='Acostic')
    axis.legend(loc='best', bbox_to_anchor=(0.35,0.77))
    axis.view_init(elev=4., azim=45)
    ##--------------
    axis.set_xlabel("$K_{x}$")
    axis.set_ylabel("$K_{y}$")
    axis.zaxis.set_rotate_label(False)  # disable automatic rotation
    axis.set_zlabel("Energy (meV)", rotation=90)
    axis.text2D(0.3, 0.79, "Magnon band structure", transform=axis.transAxes)
    ##-------------
    axis.set_xticks([round(min(x),1)+.1,0.0,round(max(x),1)-.1])
    axis.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ##---- 1BZ
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.scatter(x1, y1,z1 , color='k')
    ##--------
    figure.tight_layout()
    plt.rcParams.update({'figure.max_open_warning': 0})
    figure.savefig('Band_Structure_DMI_{}.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
print('\n----------------------------------------------\n'
      +"Calculation of Magnon Bands for DMI =  {:.3f}".format(DMI[1])
      +'\n----------------------------------------------\n')
Plot_Band_Structure3D(K_BZ,E_BZ)
#P_DOS(DOS)
if E_A_B[0] < 0.0 :
    print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n')
    print('   #########################################################################')
    print('   #Since the Energy at (\u0393) point is NEGATIVE, the calculation WILL STOP HERE#')
    print('   #########################################################################')
    print('------------------------------------------'+'\n')
    sys.exit()
print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n'
     +'E(Gap) = {:0.3f}'.format(min(E_BZ[1])-max(E_BZ[0]))+' (meV) '+'\n'
     )
print('------------------------------- '
   +'\nPlotting The Berry Phase (2D):'
   +'\n------------------------------- '+'\n') 
Be_phase_im_2D(Be_CR_Beery_plot,K_BZ_Beery_plot)
print('------------------------------- '
   +'\nPlotting The Berry Phase (3D):'
   +'\n------------------------------- '+'\n') 
Be_phase_3D(Be_CR_Beery_plot,K_BZ_Beery_plot)
##############################################
## Calculation Of Thermal Hall Conductivity
##############################################
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##-- Beery_conections_Optical
def B_C_O(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_O_star[K0],np.dot(Vy(K0),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vx(K0),psi_O[K0])))
    B_C_2 = ( np.dot(psi_O_star[K0],np.dot(Vx(K0).conj(),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vy(K0).conj(),psi_O[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_O, dataset)
        Ber_CR_O = np.array(result)
Be_CR_O = [round(num, 1) for num in Ber_CR_O]
Chern_Number_O = (sum(Be_CR_O)/(2*np.pi*K_BZ.shape[0]))
print(colored("Chern Number for Optical Mode = {:.7f}".format(Chern_Number_O)+'\n'+'\n', 'red'))
##-- Beery_conections_Acoustic
def B_C_A(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_A_star[K0],np.dot(Vy(K0),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vx(K0),psi_A[K0])))
    B_C_2 = ( np.dot(psi_A_star[K0],np.dot(Vx(K0).conj(),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vy(K0).conj(),psi_A[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_A, dataset)
        Ber_CR_A = np.array(result)
Be_CR_A = [round(num, 1) for num in Ber_CR_A]
Chern_Number_A = (sum(Be_CR_A)/(2*np.pi*K_BZ.shape[0]))
print(colored("Chern Number for Acoustic Mode = {:.7f}".format(Chern_Number_A)+'\n'+'\n', 'blue'))
#---------------------
############################################
##--    Temperature Range and steps
############################################
# intail_Temp = int(input("Enter Intail Temperature : ")) #
# Final_Temp  = int(input("Enter Final Temperature: ")) #
# steps_Temp  = int(input("Enter Steps of Temperature: ")) #
intail_Temp = 0
Final_Temp  = 310
steps_Temp  = 10
############################################
##-- Calculating The Ocupation Number & C2
############################################
import mpmath as MPM
MPM.mp.dps = 15; MPM.mp.pretty = True
print('---------------------------------------- '
   +'\nCalculating The ocupation number :'
   +'\n---------------------------------------- '+'\n')
##-- Bos-Einstin Distrbution
print(colored('-------------------\n'
     +'The Acoustic mode :'
     +'\n', 'blue'))
def nk_A(t,K): # Acostic Mode
    if t == 0.0 :
        oc_A = 0.0
    else:
        EXP_A = np.exp(np.float128((E_BZ[0,K])/(KB*t)))-1 # Exponential - 1
        oc_A = (1/EXP_A)
    ocup_A = oc_A
    return float("%.2f" %(ocup_A))
Temp =  [float("%.2f" %(i)) for i in np.arange(intail_Temp,Final_Temp,steps_Temp)]
##----- Ocupation number for Acustic mode
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_A,dataset)
n_k_A = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
##n_k_A = np.round_(n_k_A,2)
print(colored('Done.\n', 'blue'))
##----- Ocupation number for Optical mode
###
print(colored('------------------\n'
     +'The Optical mode :'
     +'\n', 'red'))
def nk_O(t,K): # Optical Mode
    if t == 0.0 :
        oc_O = 0
    else:
        EXP_O = np.exp(np.float128((E_BZ[1,K])/(KB*t)))-1 # Exponential - 1
        oc_O = (1/EXP_O)
    ocup_O = oc_O
    return float("%.2f" %(ocup_O))
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_O,dataset)
n_k_O = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
##n_k_O = np.round_(n_k_O,2)
print(colored('Done.\n', 'red'))
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(n_k_O[i,:],'r*')
#     axis[0].set_title('Ocupation for Optical mode')
#     axis[1].plot(n_k_A[i,:],'bo')
#     axis[1].set_title('Ocupation for Acustic mode')
#     plt.suptitle('Temperature_{}'.format(Temp[i]),fontsize=20)
#     fig.savefig("Ocupation_Temperature_{}.pdf".format(Temp[i]))
#     plt.show()
###############----------------------
##--- C2(nk_A)
print(colored('---------------------------------------- '
   +'\nCalculating The C2 for Acoustic mode :'
   +'\n---------------------------------------- '+'\n', 'blue'))
def C2_nk_A(x,y) :
    if n_k_A[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 = ( (1+n_k_A[x,y])*(np.log((1+n_k_A[x,y])/n_k_A[x,y])**2)
               -(np.log(n_k_A[x,y])**2)-(2*MPM.polylog(2, -n_k_A[x,y]))
              )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_A,dataset)
C2_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
##C2_A = np.round_(C2_A,2)
print(colored('Done.\n', 'blue'))
##--- C2(nk_O)
print(colored('---------------------------------------- '
   +'\nCalculating The C2 for Optical mode :'
   +'\n---------------------------------------- '+'\n', 'red'))
def C2_nk_O(x,y) :
    if n_k_O[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 =( (1+n_k_O[x,y])*(np.log((1+n_k_O[x,y])/n_k_O[x,y])**2)
             -(np.log(n_k_O[x,y])**2)-(2*MPM.polylog(2, -n_k_O[x,y]))
             )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_O,dataset)
C2_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
##C2_O = np.round_(C2_O,2)
print(colored('Done.\n', 'red'))
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_O[i,:],'r*')
#     axis[0].set_title('C2 for Optical mode')
#     axis[1].plot(C2_A[i,:],'bo')
#     axis[1].set_title('C2 for Acustic mode')
#     plt.suptitle('Temperature_{}'.format(Temp[i]),fontsize=20)
#     fig.savefig("C2_Temperature_{}.pdf".format(Temp[i]))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print(colored('---------------------------------------- '
   +'\nCalculating The C2 * BC Acoustic :'
   +'\n---------------------------------------- '+'\n', 'blue'))
##--- C2 * BC Acustic
def K_xy_A(t,K):
    C2BC_A = C2_A[t,K]*(Be_CR_A[K])
    return C2BC_A
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_A,dataset)
C2_BC_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
##C2_BC_A = np.round_(C2_BC_A,2)
print(colored('Done.\n', 'blue'))
print(colored('---------------------------------------- '
   +'\nCalculating The C2 * BC Optical :'
   +'\n---------------------------------------- '+'\n', 'red'))
##----- C2 * BC Optic
def K_xy_O(t,K):
    C2BC_O = C2_O[t,K]*(Be_CR_O[K])
    return C2BC_O
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_O,dataset)
C2_BC_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
##C2_BC_O = np.round_(C2_BC_O,2)
print(colored('Done.\n', 'red'))
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_BC_O[i,:],'r*')
#     axis[0].set_title('C2 * BC Optical mode')
#     axis[1].plot(C2_BC_A[i,:],'bo')
#     axis[1].set_title('C2 * BC Acustic mode')
#     plt.suptitle('Temperature_{}'.format(Temp[i]),fontsize=20)
#     fig.savefig("C2BC_Temperature_{}.pdf".format(Temp[i]))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print('---------------------------------------- '
   +'\nCalculating Thermal Hall Conductivity :'
   +'\n---------------------------------------- '+'\n')
K_XY_pos = []
K_XY_pos_A = []
K_XY_pos_O = []
for t in range(len(Temp)):
    start_itr_time = datetime.now()
    print('TEMPERATURE = {:.2f}'.format(Temp[t]) + '\n')
    K_XY_O = sum(C2_BC_O[t])/K_BZ.shape[0]
    K_XY_pos_O.append(K_XY_O)
    K_XY_A = sum(C2_BC_A[t])/K_BZ.shape[0]
    K_XY_pos_A.append(K_XY_A)
    cof = ((2*np.pi)**-2)*(Norm(np.cross(a,b))**-1)*(1.8)*Temp[t]
    K_XY_total = cof*(K_XY_O + K_XY_A)
    print(str(K_XY_O + K_XY_A) +' & '+str(K_XY_total))
    K_XY_pos.append(-K_XY_total) 
    ########################
    end_itr_time = datetime.now()
    print('K_XY_O = '+str(float("%.4f" %((K_XY_pos_O[-1]))))+'\n'
         +'K_XY_A = '+str(float("%.4f" %((K_XY_pos_A[-1]))))+'\n')
    print('Termal Hall Conductivity for CrBr\u2083 = '+str(float("%.4f" %((K_XY_pos[-1]))))+'\n'
         +'\n'+'Duration of each Temperature step: {}'.format(end_itr_time - start_itr_time)
         +'\n---------------------------------')
#---- Ploting THC
def Plot_TH(KXY_p): 
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(KXY_p)) - 1
    end   = int(np.max(KXY_p)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    #ax.set_xlim(0,Temp[-2])
    ax.plot(Temp,KXY_p,'r', label='DMI = {}'.format(DMI[1]))
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    ax.set_title('Hall Conductivity')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa^{xy}x10^{-12}$(W/K)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig.savefig('THC_DMI_{}_T_{}.pdf'.format(DMI[1],Temp[t]),dpi=1000)
    plt.show()
    return
# print('--------------------------------\n'
#      +'Ploting THC for Acoustic mode :'
#      +'\n--------------------------------\n')
# Plot_TH(K_XY_pos_A)
# print('Done.\n')
# print('--------------------------------\n'
#      +'Ploting THC for Optical mode :'
#      +'\n--------------------------------\n')
# Plot_TH(K_XY_pos_O)
# print('Done.\n')
print('---------------\n'
     +'Ploting THC :'
   +'\n---------------\n')
Plot_TH(K_XY_pos)

###########################################################
##-- Considering DMI x -1
###########################################################
DMI = [0.0,-0.22,0.0]
print('\n------------\n'
      +'DMI x -1 : '
      +'\n-----------\n')
##-- phase
if J[1] == 0 and DMI[1]== 0 :
    phase = 0.0
elif J[1] == 0 :
    phase =np.pi/2 # if J2 = 0.0
elif J[1] != 0 :
    phase = np.arctan(DMI[1]/(J[1]))
##--------------------------------
##  The Velocity Functions (Start)
##--------------------------------
#-- VX
##-- The velocity Site A and B
def VX_AB(K) :
    x_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        x_AB_FT.append(R_J1[q][0]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        x_AB_FT.append(R_J3[q][0]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_x_AB = np.sum(x_AB_FT)
    return V_x_AB
##-- The velocity Site A and A
def VX_AA(K) :
    x_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_AA_FT.append(R_J2[p][0]*mt.sin(phase-np.dot(K,R_J2[p])))
    V_x_AA = np.sum(x_AA_FT)
    return V_x_AA
##-- The velocity Site B and B
def VX_BB(K) :
    x_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        x_BB_FT.append(R_J2[p][0]*mt.sin(phase+np.dot(K,R_J2[p])))
    V_x_BB = np.sum(x_BB_FT)
    return V_x_BB
#-- Vy
##-- The velocity Site A and B
def VY_AB(K) :
    y_AB_FT = [] # Fourier Transform
    for q in range(len(R_J1)):
        y_AB_FT.append(R_J1[q][1]*np.exp(np.dot(K,R_J1[q])*(-1j))*J[0])
        y_AB_FT.append(R_J3[q][1]*np.exp(np.dot(K,R_J3[q])*(-1j))*J[2])
    V_y_AB = np.sum(y_AB_FT)
    return V_y_AB
##-- The velocity Site A and A
def VY_AA(K) :
    y_AA_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_AA_FT.append(R_J2[p][1]*mt.sin(phase-np.dot(K,R_J2[p])))
    V_y_AA = np.sum(y_AA_FT)
    return V_y_AA
##-- The velocity Site B and B
def VY_BB(K) :
    y_BB_FT = [] # Fourier Transform
    for p in range(len(R_J2)):
        y_BB_FT.append(R_J2[p][1]*mt.sin(phase+np.dot(K,R_J2[p])))
    V_y_BB = np.sum(y_BB_FT)
    return V_y_BB
##--------------------------------
##  The Velocity Functions (End)
##--------------------------------
#-------------------------------
## Ploting Beery Phase Colorbar
##------------------------------
#---- Sampling 1BZ
K_BZ_Beery_plot_x_limit = []
for i in K_BZ_Beery_plot:
    if (i[0] <= round(K6[0],3)+.1 and i[0] >= 0) :
        K_BZ_Beery_plot_x_limit.append(i)
    if (i[0] >= round(K3[0],3)-.1 and i[0] < 0):
        K_BZ_Beery_plot_x_limit.append(i)
K_BZ_Beery_plot_y_limit = []
for i in K_BZ_Beery_plot_x_limit:
    if (i[1] <= round(K2[1],3)+.2 and i[1] > 0 ) :
        K_BZ_Beery_plot_y_limit.append(i)
    if (i[1] >= round(K5[1],3)-.2 and i[1] < 0):
        K_BZ_Beery_plot_y_limit.append(i)
K_BZ_Beery_plot = np.zeros([len(K_BZ_Beery_plot_y_limit),3])
for i in range(K_BZ_Beery_plot.shape[0]) :
    K_BZ_Beery_plot[i] = K_BZ_Beery_plot_y_limit[i]
##----- Hamiltionian
def B_matrix_Beery_plot(K0):
    Ha_B_Beery_plot = np.zeros([2,2],dtype=complex) 
    ##############
    Ha_B_Beery_plot[0,0] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_AA(K_BZ_Beery_plot[K0]))
                -(2*Spin*J_tilda)-(2*Spin*A)+(g*mu_B)*B )
    ##############
    Ha_B_Beery_plot[0,1] = 2*Spin*J_K_AB(K_BZ_Beery_plot[K0])
    ##############
    Ha_B_Beery_plot[1,0] = np.conjugate(Ha_B_Beery_plot[0,1])
    ##############   
    Ha_B_Beery_plot[1,1] = (4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(J_K_BB(K_BZ_Beery_plot[K0]))-(2*Spin*J_tilda)
                -(2*Spin*A)+(g*mu_B)*B)
    H_B_Beery_plot = Ha_B_Beery_plot
    return H_B_Beery_plot #-- Hamiltonian of LSWT
###--- EigenValues function for K points
E_A_B_Beery_plot = []
psi_A_Beery_plot = []
psi_A_star_Beery_plot = []
E_O_B_Beery_plot = []
psi_O_Beery_plot = []
psi_O_star_Beery_plot = []
###--- Wave function for K points
def PSI_A_O_Beery_plot(kpoint):
    v_L_B, w_L_B = eigh(B_matrix_Beery_plot(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result_Beery_plot = pool.map(PSI_A_O_Beery_plot, dataset)
        PSI_L_Beery_plot = result_Beery_plot
for i in range(K_BZ_Beery_plot.shape[0]):
    E_A_B_Beery_plot.append(PSI_L_Beery_plot[i][0][0])
    psi_A_Beery_plot.append(PSI_L_Beery_plot[i][1][0])
    psi_A_star_Beery_plot.append(PSI_L_Beery_plot[i][1][0].reshape(1,2).conj())
    ########
    E_O_B_Beery_plot.append(PSI_L_Beery_plot[i][0][1])
    psi_O_Beery_plot.append(PSI_L_Beery_plot[i][1][1])
    psi_O_star_Beery_plot.append(PSI_L_Beery_plot[i][1][1].reshape(1,2).conj())
E_BZ_Beery_plot = np.zeros([N_M_S,K_BZ_Beery_plot.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ_Beery_plot[0,i] = E_A_B_Beery_plot[i]
    E_BZ_Beery_plot[1,i] = E_O_B_Beery_plot[i]
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ_Beery_plot[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ_Beery_plot[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ_Beery_plot[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##----- Calculating the Chern Numbers
##-- Berry Function
def Beery_conections_plot(K0):
    beery_cof = 1./((E_O_B_Beery_plot[K0]-E_A_B_Beery_plot[K0])**2)
    B_C_1_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vy(K0),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vx(K0),psi_O_Beery_plot[K0])))
    B_C_2_plot = ( np.dot(psi_O_star_Beery_plot[K0],np.dot(Vx(K0).conj(),psi_A_Beery_plot[K0]))
             *np.dot(psi_A_star_Beery_plot[K0],np.dot(Vy(K0).conj(),psi_O_Beery_plot[K0])))
    B_C_3_plot = beery_cof*(B_C_1_plot+B_C_2_plot)
    return B_C_3_plot[0].imag
Be_CR = 0
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ_Beery_plot.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(Beery_conections_plot, dataset)
        Ber_CR_Beery_plot = np.array(result)
Be_CR_Beery_plot = [round(num, 1) for num in Ber_CR_Beery_plot]
def Be_phase_im_2D(Beery,K):
    # convert to arrays to make use of previous answer to similar question
    x = K[:,0]
    y = K[:,1]
    z = np.asarray(Beery)

    # Set up a regular grid of interpolation points
    nInterp = 400
    xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(), y.max(), nInterp)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate; there's also method='cubic' for 2-D data such as here
    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

    img = plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower', interpolation='nearest',
                     cmap='coolwarm',extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto') 
    plt.xlim(x.min()-.02,x.max()+.02)
    plt.ylim(y.min()-.03,y.max()+.03)
    plt.xlabel("$K_{x}$")
    plt.ylabel("$K_{y}$")        
    cbar = plt.colorbar(img)
    cbar.ax.get_yaxis().labelpad = 7
    cbar.ax.set_ylabel('Beery cervature', rotation=90)
    plt.savefig('Berry_phase_DMII_{}_2D.pdf'.format(DMI[1]),dpi=1200,bbox_inches="tight")
    plt.show()
    return
#---------------------
def Be_phase_3D(Berry,K):
    X = K[:,0]
    Y = K[:,1]
    Z = np.array([(Berry),(Berry)])
    # Normalize to [0,1]
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.coolwarm(norm(Z))
    rcount, ccount, _ = colors.shape
    zlim = (0,0.1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    #--------------------------------
    ax.view_init(elev=25., azim=45)
    ax.set_xlabel("$K_{x}$")
    ax.set_ylabel("$K_{y}$")
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("Berry phase", rotation=90)
    ##-------------
    ax.set_xticks([round(min(X),1)+0.2,0.0,round(max(X),1)-0.2])
    ax.set_yticks([round(min(Y),1)+0.2,0.0,round(max(Y),1)-0.2])
    ##--------------
    fig.savefig('Berry_phase_DMII_{}_3D.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
#------------------------------------------------------------------
##-- Calculating Beery cervature & Chern number
print('---------------------------------------------------------------- '
   +'\nCalculating Beery cervature & Chern number (General Formulas)  :'
   +'\n---------------------------------------------------------------- '+'\n') 
###--- EigenValues function for K points
E_A_B = []
psi_A = []
psi_A_star = []
E_O_B = []
psi_O = []
psi_O_star = []
###--- Wave function for K points
def PSI_A_O(kpoint):
    v_L_B, w_L_B = eigh(B_matrix(kpoint))
    return v_L_B,w_L_B
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(PSI_A_O, dataset)
        PSI_L = result
for i in range(K_BZ.shape[0]):
    E_A_B.append(PSI_L[i][0][0])
    psi_A.append(PSI_L[i][1][0])
    psi_A_star.append(PSI_L[i][1][0].reshape(1,2).conj())
    ########
    E_O_B.append(PSI_L[i][0][1])
    psi_O.append(PSI_L[i][1][1])
    psi_O_star.append(PSI_L[i][1][1].reshape(1,2).conj())
E_BZ = np.zeros([N_M_S,K_BZ.shape[0]])
for i in range(E_BZ.shape[1]):
    E_BZ[0,i] = E_A_B[i]
    E_BZ[1,i] = E_O_B[i]
##------------------------------------------
##   Function for Magnon-DOS Plots (Start)
##------------------------------------------
DOS = E_A_B
for i in E_O_B:
    DOS.append(i)
##-- now plot density of states
def P_DOS(E3):
    fig, ax = plt.subplots()
    ax.hist(E3,80,range=(np.min(E3),np.max(E3)))
    ax.set_ylim(0.0,100.0)
    # put title
    ax.set_title("Density of states")
    ax.set_xlabel("Band energy")
    ax.set_ylabel("Number of states")
    # make an PDF figure of a plot
    fig.tight_layout()
    #fig.savefig("Inversion-symmetry_dos_CrBr3.pdf")
    plt.show()
def Plot_Band_Structure3D(K,E): # E1 Energy for Acoustic mode & E2 for Optical mode
    figure = plt.figure(figsize=(6,5))
    axis = figure.add_subplot(111, projection = '3d')
    ##-----------------
    x = K[:,0]
    y = K[:,1]
    en_ac = np.array([E[0,:],E[0,:]])
    en_op = np.array([E[1,:],E[1,:]])
    ##----------------
    # axis.scatter(x, y, c='k', marker='*')
    axis.plot_wireframe(x, y, en_op, cstride=1, color='r', label='Optical')
    axis.plot_wireframe(x, y, en_ac, rstride=1, color='b', label='Acostic')
    axis.legend(loc='best', bbox_to_anchor=(0.35,0.77))
    axis.view_init(elev=4., azim=45)
    ##--------------
    axis.set_xlabel("$K_{x}$")
    axis.set_ylabel("$K_{y}$")
    axis.zaxis.set_rotate_label(False)  # disable automatic rotation
    axis.set_zlabel("Energy (meV)", rotation=90)
    axis.text2D(0.3, 0.79, "Magnon band structure", transform=axis.transAxes)
    ##-------------
    axis.set_xticks([round(min(x),1)+.1,0.0,round(max(x),1)-.1])
    axis.set_yticks([round(min(y),1),0.0,round(max(y),1)])
    ##---- 1BZ
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross],
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.plot_wireframe(x1, y1,z1 , color='c',linestyle='dashed')
    ##---- K points
    x1 = np.array([round(K1[0], 3),round(K2[0], 3),round(K3[0], 3),
                   round(K4[0], 3),round(K5[0], 3),round(K6[0], 3),round(K1[0], 3)])
    y1 = np.array([round(K1[1], 3),round(K2[1], 3),round(K3[1], 3),
                   round(K4[1], 3),round(K5[1], 3),round(K6[1], 3),round(K1[1], 3)])
    z1 = np.array([
        [en_cross,en_cross,en_cross,en_cross,en_cross,en_cross,en_cross]
             ])
    axis.scatter(x1, y1,z1 , color='k')
    ##--------
    figure.tight_layout()
    plt.rcParams.update({'figure.max_open_warning': 0})
    figure.savefig('Band_Structure_DMI_{}.pdf'.format(DMI[1]),dpi=2**20)
    plt.show()
    return
print('\n----------------------------------------------\n'
      +"Calculation of Magnon Bands for DMI =  {:.3f}".format(DMI[1])
      +'\n----------------------------------------------\n')
Plot_Band_Structure3D(K_BZ,E_BZ)
#P_DOS(DOS)
if E_A_B[0] < 0.0 :
    print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n')
    print('   #########################################################################')
    print('   #Since the Energy at (\u0393) point is NEGATIVE, the calculation WILL STOP HERE#')
    print('   #########################################################################')
    print('------------------------------------------'+'\n')
    sys.exit()
print('E(\u0393) = {:0.3f}'.format(E_A_B[0])+' (meV) '+'\n'
     +'E(Gap) = {:0.3f}'.format(min(E_BZ[1])-max(E_BZ[0]))+' (meV) '+'\n'
     )
print('------------------------------- '
   +'\nPlotting The Berry Phase (2D):'
   +'\n------------------------------- '+'\n') 
Be_phase_im_2D(Be_CR_Beery_plot,K_BZ_Beery_plot)
print('------------------------------- '
   +'\nPlotting The Berry Phase (3D):'
   +'\n------------------------------- '+'\n') 
Be_phase_3D(Be_CR_Beery_plot,K_BZ_Beery_plot)
##############################################
## Calculation Of Thermal Hall Conductivity
##############################################
def Vx(K0):
    Ha_X= np.zeros([2,2],dtype=complex)
    ##############
    Ha_X[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_AA(K_BZ[K0]))
    ##############
    Ha_X[0,1] = 2*Spin*(-1j)*(VX_AB(K_BZ[K0]))
    ##############
    Ha_X[1,0] = np.conjugate(Ha_X[0,1])
    ##############
    Ha_X[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VX_BB(K_BZ[K0]))
    H_X = Ha_X
    return H_X   #-- Hamiltonian of Velocity operators  in X direction
def Vy(K0):
    Ha_Y= np.zeros([2,2],dtype=complex)
    #############
    Ha_Y[0,0] = 4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_AA(K_BZ[K0]))
    ##############
    Ha_Y[0,1] = 2*Spin*(-1j)*(VY_AB(K_BZ[K0]))
    ##############
    Ha_Y[1,0] = np.conjugate(Ha_Y[0,1])
    ##############
    Ha_Y[1,1] = -4*Spin*np.sqrt(J[1]**2+DMI[1]**2)*(VY_BB(K_BZ[K0]))
    H_Y = Ha_Y
    return H_Y   #-- Hamiltonian of Velocity operators  in X direction
##-- Beery_conections_Optical
def B_C_O(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_O_star[K0],np.dot(Vy(K0),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vx(K0),psi_O[K0])))
    B_C_2 = ( np.dot(psi_O_star[K0],np.dot(Vx(K0).conj(),psi_A[K0]))
             *np.dot(psi_A_star[K0],np.dot(Vy(K0).conj(),psi_O[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_O, dataset)
        Ber_CR_O = np.array(result)
Be_CR_O = [round(num, 1) for num in Ber_CR_O]
Chern_Number_O = (sum(Be_CR_O)/(2*np.pi*K_BZ.shape[0]))
print(colored("Chern Number for Optical Mode = {:.7f}".format(Chern_Number_O)+'\n'+'\n', 'red'))
##-- Beery_conections_Acoustic
def B_C_A(K0):
    beery_cof = 1./((E_O_B[K0]-E_A_B[K0])**2)
    B_C_1 = ( np.dot(psi_A_star[K0],np.dot(Vy(K0),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vx(K0),psi_A[K0])))
    B_C_2 = ( np.dot(psi_A_star[K0],np.dot(Vx(K0).conj(),psi_O[K0]))
             *np.dot(psi_O_star[K0],np.dot(Vy(K0).conj(),psi_A[K0])))
    B_C_3 = beery_cof*(B_C_1+B_C_2)
    return B_C_3[0].imag
##-- Calculating the Chern Numbers
if __name__ == '__main__':
    dataset = [i for i in range(0,K_BZ.shape[0],1)]
    cpus = mp.cpu_count()
    with Pool(processes= cpus) as pool:
        result = pool.map(B_C_A, dataset)
        Ber_CR_A = np.array(result)
Be_CR_A = [round(num, 1) for num in Ber_CR_A]
Chern_Number_A = (sum(Be_CR_A)/(2*np.pi*K_BZ.shape[0]))
print(colored("Chern Number for Acoustic Mode = {:.7f}".format(Chern_Number_A)+'\n'+'\n', 'blue'))
############################################
##-- Calculating The Ocupation Number & C2
############################################
import mpmath as MPM
MPM.mp.dps = 15; MPM.mp.pretty = True
print('---------------------------------------- '
   +'\nCalculating The ocupation number :'
   +'\n---------------------------------------- '+'\n')
##-- Bos-Einstin Distrbution
def nk_A(t,K): # Acostic Mode
    if t == 0.0 :
        oc_A = 0.0
    else:
        EXP_A = np.exp(np.float128((E_BZ[0,K])/(KB*t)))-1 # Exponential - 1
        oc_A = (1/EXP_A)
    ocup_A = oc_A
    return float("%.6f" %(ocup_A))
##----------------------------
Temp =  [float("%.2f" %(i)) for i in np.arange(intail_Temp,Final_Temp,steps_Temp)]
##----- Ocupation number for Acustic mode
print(colored('---------------------------------------- '
   +'\nThe Acoustic mode :'
   +'\n', 'blue'))
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_A,dataset)
n_k_A = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
print(colored('Done.\n', 'blue'))
##----- Ocupation number for Optical mode
###
def nk_O(t,K): # Optical Mode
    if t == 0.0 :
        oc_O = 0
    else:
        EXP_O = np.exp(np.float128((E_BZ[1,K])/(KB*t)))-1 # Exponential - 1
        oc_O = (1/EXP_O)
    ocup_O = oc_O
    return float("%.6f" %(ocup_O))
print(colored('---------------------------------------- '
   +'\nThe Optical mode :'
   +'\n', 'red'))
if __name__ == '__main__':
    dataset = [(i,j) for i in Temp for j in range(0,K_BZ.shape[0],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(nk_O,dataset)
n_k_O = (np.array(result).reshape(len(Temp),E_BZ.shape[1]))
print(colored('Done.\n', 'red'))
##---------------------------
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(n_k_O[i,:],'r*')
#     axis[0].set_title('Ocupation for Optical mode')
#     axis[1].plot(n_k_A[i,:],'bo')
#     axis[1].set_title('Ocupation for Acustic mode')
#     #fig.savefig("Temperature_{}.pdf".format(i))
#     plt.show()
###############----------------------
##--- C2(nk_A)
print(colored('---------------------------------------- '
   +'\nCalculating The C2 for Acoustic mode :'
   +'\n---------------------------------------- '+'\n', 'blue'))
def C2_nk_A(x,y) :
    if n_k_A[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 = ( (1+n_k_A[x,y])*(np.log((1+n_k_A[x,y])/n_k_A[x,y])**2)
               -(np.log(n_k_A[x,y])**2)-(2*MPM.polylog(2, -n_k_A[x,y]))
              )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_A,dataset)
C2_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
print(colored('Done.\n', 'blue'))
##--- C2(nk_O)
print(colored('---------------------------------------- '
   +'\nCalculating The C2 for Optical mode :'
   +'\n---------------------------------------- '+'\n', 'red'))
def C2_nk_O(x,y) :
    if n_k_O[x,y] == 0.0 : # (np.log((1+x)/x) when x---->0 == 1 & np.log(x) = inf
        ##b/c the coeffiecnt t at zero temperature is absolut zero and this inf number will be multply
        ##by ablsolute zero, Since any thing in absolut zero is zero, we can consider it zero
        c_2 = 0.0    
    else:
        c_2 =( (1+n_k_O[x,y])*(np.log((1+n_k_O[x,y])/n_k_O[x,y])**2)
             -(np.log(n_k_O[x,y])**2)-(2*MPM.polylog(2, -n_k_O[x,y]))
             )
    return c_2
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(C2_nk_O,dataset)
C2_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
print(colored('Done.\n', 'red'))
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_O[i,:],'r*')
#     axis[0].set_title('C2 for Optical mode')
#     axis[1].plot(C2_A[i,:],'bo')
#     axis[1].set_title('C2 for Acustic mode')
#     #fig.savefig("Temperature_{}.pdf".format(i))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print(colored('---------------------------------------- '
   +'\nCalculating The C2 * BC Acoustic :'
   +'\n---------------------------------------- '+'\n', 'blue'))
##--- C2 * BC Acustic
def K_xy_A(t,K):
    C2BC_A = C2_A[t,K]*(Be_CR_A[K])
    return C2BC_A
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_A.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_A,dataset)
C2_BC_A = (np.array(result).reshape(len(Temp),n_k_A.shape[1]))
print(colored('Done.\n', 'blue'))
print(colored('---------------------------------------- '
   +'\nCalculating The C2 * BC Optical :'
   +'\n---------------------------------------- '+'\n', 'red'))
##----- C2 * BC Optic
def K_xy_O(t,K):
    C2BC_O = C2_O[t,K]*(Be_CR_O[K])
    return C2BC_O
if __name__ == '__main__':
    dataset = [(i,j) for i in range(len(Temp)) for j in range(0,n_k_O.shape[1],1)]
    result = 0.0
    with Pool(processes=cpus) as pool:
        result = pool.starmap(K_xy_O,dataset)
C2_BC_O = (np.array(result).reshape(len(Temp),n_k_O.shape[1]))
print(colored('Done.\n', 'red'))
# for i in range(len(Temp)):
#     print('\n-------------------\n Temperature = ' + str(Temp[i])+'\n-------------------\n')
#     fig, axis = plt.subplots(1,2)
#     axis[0].plot(C2_BC_O[i,:],'r*')
#     axis[0].set_title('C2 * BC Optical mode')
#     axis[1].plot(C2_BC_A[i,:],'bo')
#     axis[1].set_title('C2 * BC Acustic mode')
#     #fig.savefig("C2BC_Temperature_{}.pdf".format(i))
#     plt.show()
#########################################################
##-- Calculating Thermal Hall Conductivity
#########################################################
print('---------------------------------------- '
   +'\nCalculating Thermal Hall Conductivity :'
   +'\n---------------------------------------- '+'\n')
K_XY_neg = []
K_XY_neg_A = []
K_XY_neg_O = []
for t in range(len(Temp)):
    start_itr_time = datetime.now()
    print('TEMPERATURE = {:.2f}'.format(Temp[t]) + '\n')
    K_XY_O = sum(C2_BC_O[t])/K_BZ.shape[0]
    K_XY_neg_O.append(K_XY_O)
    K_XY_A = sum(C2_BC_A[t])/K_BZ.shape[0]
    K_XY_neg_A.append(K_XY_A)
    cof = ((2*np.pi)**-2)*(Norm(np.cross(a,b))**-1)*(1.8)*Temp[t]
    K_XY_total = cof*(K_XY_O + K_XY_A)
    K_XY_neg .append(-K_XY_total) 
    ########################
    end_itr_time = datetime.now()
    print('K_XY_O = '+str(float("%.4f" %((K_XY_neg_O[t]))))+'\n'
         +'K_XY_A = '+str(float("%.4f" %((K_XY_neg_A[t]))))+'\n')
    print('Termal Hall Conductivity for CrBr\u2083 = '+str(float("%.4f" %((K_XY_neg[t]))))+'\n'
         +'\n'+'Duration of each Temperature step: {}'.format(end_itr_time - start_itr_time)
         +'\n---------------------------------')
#---- Ploting THC
def Plot_TH_N(KXY_n): 
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(KXY_n)) - 1
    end   = int(np.max(KXY_n)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    #ax.set_xlim(0,Temp[-2])
    ax.plot(Temp,KXY_n,'b', label='DMI = {}'.format(DMI[1]))
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    ax.set_title('Hall Conductivity is divided by area')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa^{xy}x10^{-12}$(W/K)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    fig.savefig('THC_DMI_{}_T_{}.pdf'.format(DMI[1],Temp[t]),dpi=1000)
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.show()
    return
Plot_TH_N(K_XY_neg)
print('-----------------------\n'
     +'Ploting THC for Both :'
   +'\n-----------------------\n')
#---- Ploting THC
def Plot_TH_T(KXY_p,KXY_n): 
    fig, ax = plt.subplots(dpi=100) #,figsize = (4,4)
    ##- specify horizontal axis details
    start = int(np.min(KXY_p)) - 1
    end   = int(np.max(KXY_p)) + 1
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    #ax.set_xlim(0,Temp[-2])
    ax.plot(Temp,KXY_p,'r', label='DMI = {}'.format(-DMI[1]))
    ax.plot(Temp,KXY_n,'b', label='DMI = {}'.format(DMI[1]))
    #-- Set up grid, legend
    ax.legend(frameon=False)
    ##-- put title
    ax.set_title('Hall Conductivity is not divided by area')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa^{xy}x10^{-12}$(W/K)")
    plt.grid(True,linestyle = ':',alpha=0.8)
    ##-- make an PDF figure of a plot
    fig.tight_layout()
    fig.savefig('THC_both_T_{}.pdf'.format(Temp[t]),dpi=1000)
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.show()
    return
Plot_TH_T(K_XY_pos,K_XY_neg)


# In[ ]:





# In[ ]:





# In[ ]:




