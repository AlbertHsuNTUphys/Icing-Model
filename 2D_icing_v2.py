import numpy as np
import matplotlib.pyplot as plt
import random
import copy

kB = 1.

def Monte_Carlo(status, energy, J, T):
    H = energy
    S = copy.deepcopy(status)
    N = len(status)
    steps = 200
    Hmax = abs(-1. * J * 4 * N * N)
    Mmax = float(N * N)

    H_list = np.zeros(steps//2)
    M_list = np.zeros(steps//2)
    n = 0
    if(T==0.):
        return S, H, H/Hmax, np.sum(S)/Mmax
    B = 1./(kB*T)
    for i in range(steps):
        for j in range(N):
           for k in range(N): 
               x = random.randint(0,N-1)
               y = random.randint(0,N-1)
               deltaH = 2*J*(S[x][y]*(S[(x+1)%N][y] + S[(x-1)%N][y] + S[x][(y-1)%N] + S[x][(y+1)%N]))
               if(deltaH < 0):
                   S[x][y] = -1*S[x][y]
                   H += deltaH
               else:
                   threshold = np.exp(-1.*B*deltaH)
                   n = random.random()
                   if(n<threshold):
                       S[x][y] = -1*S[x][y]
                       H += deltaH
        if(i >= steps//2):
            H_list[i-steps//2] = H
            M_list[i-steps//2] = np.sum(S)
            n+=1
#        if(i-steps//2>0 and abs((H_list[i-steps//2]-H_list[i-steps//2-1])/H) < 0.01):
#            continue
    return S,H,np.average(H_list)/Hmax,abs(np.average(M_list))/Mmax


def Monte_Carlo_honeycube(status, energy, J, T): 
    H = energy
    S = copy.deepcopy(status)
    N = len(status)
    steps = 200
    Hmax = abs(-1. * J * 6 * N * N)
    Mmax = float(N * N)

    H_list = np.zeros(steps//2)
    M_list = np.zeros(steps//2)
    n = 0 
    if(T==0.):
        return S, H, H/Hmax, np.sum(S)/Mmax
    B = 1./(kB*T)
    for i in range(steps):
        for j in range(N):
           for k in range(N): 
               x = random.randint(0,N-1)
               y = random.randint(0,N-1)
               deltaH = -1
               if x%2 ==1:
                   deltaH = 2*J*(S[x][y]*(S[(x+1)%N][(y-1)%N] + S[(x+1)%N][y] + S[(x-1)%N][(y-1)%N] + S[(x-1)%N][y] + S[x][(y-1)%N] + S[x][(y+1)%N]))
               else:
                   deltaH = 2*J*(S[x][y]*(S[(x+1)%N][(y+1)%N] + S[(x+1)%N][y] + S[(x-1)%N][(y+1)%N] + S[(x-1)%N][y] + S[x][(y-1)%N] + S[x][(y+1)%N]))
               if(deltaH < 0): 
                   S[x][y] = -1*S[x][y]
                   H += deltaH
               else:
                   threshold = np.exp(-1.*B*deltaH)
                   n = random.random()
                   if(n<threshold):
                       S[x][y] = -1*S[x][y]
                       H += deltaH
        if(i >= steps//2):
            H_list[i-steps//2] = H 
            M_list[i-steps//2] = np.sum(S)
            n+=1
#        if(i-steps//2>0 and abs((H_list[i-steps//2]-H_list[i-steps//2-1])/H) < 0.01):
#            continue
    return S,H,np.average(H_list)/Hmax,abs(np.average(M_list))/Mmax

            

N = 100
J = 1
n = 100
maxx = 8
dt = float(maxx)/float(n)
# Square
status = np.ones((N,N))
current_energy = -1. * J * 4 * N * N

x = np.linspace(0,maxx,n)
y1 = np.ones(n)
y2 = np.ones(n)
y3 = np.zeros(n)

for i in range(n):
  T = float(i)*float(maxx)/float(n)
  status, current_energy,H_avg,M_avg = Monte_Carlo(status,current_energy,J,T)
  print(H_avg)
  y1[i] = H_avg
  y2[i] = M_avg
  if(i > 0):
     y3[i] = (y1[i]-y1[i-1])/float(maxx)*float(n)

# Honey Cube

status = np.ones((N,N))
current_energy = -1. * J * 6 * N * N

y1_1 = np.ones(n)
y2_1 = np.ones(n)
y3_1 = np.zeros(n)

for i in range(n):
  T = float(i)*float(maxx)/float(n)
  status, current_energy,H_avg,M_avg = Monte_Carlo_honeycube(status,current_energy,J,T)
  y1_1[i] = H_avg
  y2_1[i] = M_avg
  print(H_avg)
  if(i > 0):
     y3_1[i] = (y1_1[i]-y1_1[i-1])/float(maxx)*float(n)


TC_square = np.argmax(y3)*dt
TC_honeycube = np.argmax(y3_1)*dt

print(TC_square)
print(TC_honeycube)

fig, ax = plt.subplots(2,2)

ax[0,0].plot(x,y1,'.',label='square')
ax[0,0].plot(x,y1_1,'*',label='honeycube')
ax[0,0].legend(loc='right')
ax[0,0].set_title("H")
ax[0,0].set_xlabel("T(K)")
ax[0,0].set_ylabel("H/|H(max)|")

ax[0,1].plot(x,y2,'.',label='square')
ax[0,1].plot(x,y2_1,'*',label='honeycube')
ax[0,1].legend(loc='right')
ax[0,1].set_title("M")
ax[0,1].set_xlabel("T(K)")
ax[0,1].set_ylabel("|M/M(max)|")

ax[1,0].plot(x,y3,'.',label='square')
ax[1,0].plot(x,y3_1,'*',label='honeycube')
ax[1,0].legend(loc='right')
ax[1,0].set_title("Specific Heat")
ax[1,0].set_xlabel("T(K)")
ax[1,0].set_ylabel("d(H/|H(max)|)/dt")
plt.show()

