import matplotlib.pyplot as plt #General Plotting
from matplotlib import cm

import numpy as np

from scipy.stats import multivariate_normal

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(1)

# Number of samples to draw from each distribution
N = 10000

p0 = 0.65 # P(L=0)
p1 = 0.35 # P(L=1)

N0=0
N1=0


for i in range(0,N):
    if np.random.random() < p0:
        N0= N0 + 1
    else:
        N1 = N1 + 1

# mu and sigma values for the data distribution 0
mu0 = np.transpose(np.array([-1/2, -1/2, -1/2]))
sigma0 = np.array([[1,-0.5,0.3],[-0.5,1,-0.5],[0.3, -0.5, 1]])

# generates data from data distribution 0 and adds it to the scatter plot
r0 = np.random.multivariate_normal(mu0,sigma0,N0)

plt.clf()
fig = plt.figure(figsize=(16, 16))
ax1 = fig.add_subplot(131, projection = '3d')
plt.xlabel("x")
plt.ylabel("y")
plt.ylabel("z")
plt.title("Full Covariance Matrix")
fig = plt.figure(0)

ax1.scatter(r0[:,0],r0[:,1],r0[:,2])

# mu and sigma values for the data distribution 1
mu1 = np.transpose(np.array([1, 1, 1]))
sigma1 = np.array([[1, 0.3, -0.2],[0.3, 1, 0.3],[-0.2, 0.3, 1]])

# generates data from data distribution 1 and adds it to the scatter plot
r1 = np.random.multivariate_normal(mu1, sigma1, N1)
ax1.scatter(r1[:,0],r1[:,1],r1[:,2],c='r')
ax1.set_xlabel("x")
ax1.set_ylabel("x")
ax1.set_zlabel("x")

ax1.set_title('True Labels')





N0_discriminant  =[]
N1_discriminant =[]
# calculates discr imi nanscor esforalld atapoints in eac h d ata distrib ution
for j in range(0, N0):
    N0_discriminant.append(multivariate_normal.pdf(r0[j], mu1, sigma1)/multivariate_normal)
for j in range(0, N1):
    N1_discriminant.append(multivariate_normal.pdf(r1[j], mu1, sigma1)/multivariate_normal) # CUT OFF
    
false_positive =[]
true_positive =[]
prob_error =[]
gamma_values =[]

full_discriminant= N0_discriminant + N1_discriminant
full_discriminant.append(0)

# Inplac eofte sti n g val u esfrom 0toin fin ity whichis impos sible, Ite stfor val u esb e twe 8
#ac ros sth e two discr imi nants cor e lists(p lusa0 valu e), soin eac h loopone new val u ei# b e g in >= 0 andar e e v e ntually <=th e maximum value
for i in sorted(full_discriminant):
fp = len([ j for j in N0_discriminant if j >=i ])/ N0
tp = len([ j for j in N1_discriminant if j >=i ])/ N1

false_positive.append(fp)
true_positive.append(tp)    
gamma_values.append(i)
prob_error.append(fp *p0 + (1 - tp)* p1)


# g etsth e minimum_error and ind e xofth e minimum_error(toplott h epointwi th)
minimum_error = min( prob_error)
minimum_index = 0
for i in range(0, len(prob_error)) :
    if prob_error [i] == minimum_error:
        minimum_index = i
        break

print('PartA')
print('Experimental Gamma ')
print(gamma_values [minimum index ])
print('Experimental min_error ')
print (min( prob_error))
print('Theor eti cal Eror ')

theo_fp = len([ j for j in N0_discriminant if j >=(p1/p0) ])/ N0
theo_tp = len([ j for j in N1_discriminant if j >=(p1/p0) ])/ N1

print(theo_fp *p0 + (1- theo_tp)* p1)

# plott h e d ata wi thth e twoad d itional d atapoints
plt.figure(1)
plt.plot(false_positive,true_positive,label= 'ROC Curve ')
plt.plot(false_positive [minimum index],tr u e_positive [minimum index], ' ro ',label= ' Experimental' )
plt.plot(theo_fp, theo_tp, 'g+ ',label= ' Theor eti cal Minimum Eror ')
plt.title('Minimum Expected Risk ROC Curve ')
plt.ylabel('P( Cor rectDetection)')
plt.xlabel('P( False Positiv e)')
plt.legend()
plt.show()




# Pa rtB :
#id e ntity matrix us ed for sigma
sigma_nb = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1] ])
N0_discriminant = []
N1_discriminant = []
# calculati n g discr imi nants cor esasa bov e
for j in range(0, N0):
    N0_discriminant.append(multivariate_normal.pdf(r0[j], mu1, sigma_nb)/multivariate_normal)
for j in range(0, N1):
    N1_discriminant.append(multivariate_normal.pdf(r1[j], mu1, sigma_nb)/multivariate_normal )

false_positive =[]
true_positive =[]
prob_error =[]
gamma_values =[]

full_discriminant= N0_discriminant + N1_discriminant
full_discriminant.append(0)
# d e cidin g forallpos sible gammas us ingth e same logic aspartA
for i in sorted(full_discriminant):
    fp = len([ j for j in N0_discriminant if j >=i ])/ N0
    tp = len([ j for j in N1_discriminant if j >=i ])/ N1
    false_positive.append(fp)
    true_positive.append(tp)
    gamma_values.append(i)
    prob_error.append(fp *p0 + (1 - tp)* p1)

minimum_error = min( prob_error)
minimum_index = 0
for i in range(0, len(prob_error)) :
    if prob_error [i] == minimum_error :
        minimum_index = i
        break
print('PartB ')
print('Experimental Gamma ')
print(gamma_values [minimum_index ])
print('Experimental min_error ')
print (min( prob_error))
print('Theor eti cal Eror ')
theo_fp = len([ j for j in N0_discriminant if j >=(p1/p0) ])/ N0
theo_tp = len([ j for j in N1_discriminant if j >=(p1/p0) ])/ N1
print(theo_fp *p0 + (1- theo_tp)* p1)
# plott h e d ata
plt.figure(2)
plt.plot(false_positive,true_positive,label= 'ROC Curve ')
plt.plot(false_positive[minimum_index],true_positive[minimum_index], ' ro ',label= ' Experimental ')
plt.plot(theo_fp, theo_tp, 'g+ ',label= ' Theor eti cal Minimum Eror ')
plt.title('Naive Bayesian ROC Curve ')
plt.ylabel('P( Cor rectDetection)')
plt.xlabel('P( False Positiv e)')
plt.legend()
plt.show()





# Pa rtC
#simple f u n ctionto g eta v e rag eofalis t
def get_average(list):
    return sum( list)/ len(list)

# calculate muproj e ction s
mu0proj = np.transpose(np.array([get_average(r0[:, 0]), get_average(r0[:, 1]), get_average(r0[:, 2])]))
mu1proj = np.transpose(np.array([get_average(r1[:, 0]), get_average(r1[:, 1]), get_average(r1[:, 2])]))# calculate covaria nceproj e ction s
sigma0proj = np.cov(r0, rowvar=False)
sigma1proj = np.cov(r1, rowvar=False)
# calculate b e twe ensc att e r and with insc att e r
Sb = (mu0proj - mu1proj)* np.transpose (mu0proj - mu1proj)
Sw = sigma0proj + sigma1proj
# g eteig e n v e cto rsand eig e n val u esfrom Swˆ-1 * Sb
w, v = np.linalg.eig(np.linalg.inv(Sw) * Sb)
max_eigen_index = 0
# fin d ind e xof maximum eig e n val u e
for i in range(0, len (w)):
    if w[i] == max(w) :
        max_eigen_index = i
# assig n wLDAto cor r e cteig e n v e cto r and calculate LDA for 2 distrib ution s
# Note : for some r eason numpy hasb othth e wLDA andth e d ataal r ead ytranspos e dsothis fix # formul a
wLDA = v[:, max_eigen_index]
yLDA0 = np.matmul(wLDA, r0.T)
yLDA1 = np.matmul(wLDA, r1.T)
totalLDA = list(yLDA0) + list(yLDA1)
false_positive =[]
true_positive =[]
prob_error =[]
gamma_values =[]
# us e same logic asa bov eto make d e cisio n s
for i in sorted(totalLDA):
    fp = len([ j for j in yLDA0 if j >=i ])/ N0
    tp = len([ j for j in yLDA1 if j >=i ])/ N1
    false_positive.append(fp)
    true_positive.append(tp)
    gamma_values.append(i)
    prob_error.append(fp *p0 + (1 - tp)* p1)

minimum_error = min( prob_error)
minimum_index = 0
for i in range(0, len(prob_error)) :
    if prob_error [i] == minimum_error :
        minimum_index = i
        break

print('PartC')
print('Experimental Gamma ')
print(gamma_values[minimum_index])
print('Experimental min_error ')
print (min( prob_error))
print('Theor eti cal Eror ')
theo_fp = len([ j for j in yLDA0 if j >=(p1/p0) ])/ N0
theo_tp = len([ j for j in yLDA1 if j >=(p1/p0) ])/ N1
print(theo_fp *p0 + (1- theo_tp)* p1)

# plotd ata
plt.figure(3)
plt.plot(false_positive,true_positive,label= 'ROC Curve ')
plt.plot(false_positive [minimum_index],true_positive [minimum_index], 'ro',label= ' Experimental ')
plt.plot(theo_fp, theo_tp, 'g+',label= ' Theor eti cal Minimum Eror ')
plt.title('Fische r LDA ROC Curve ')
plt.ylabel('P( Cor rectDetection)')
plt.xlabel('P( False Positiv e)')
plt.legend()
plt.show()