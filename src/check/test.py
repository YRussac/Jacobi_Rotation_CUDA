import math
import numpy as np

theta = [0, math.pi / 4, 2* math.pi /3]
p = [2,3,1]
q = [1,2,3]

def rotation_matrix(theta, p, q):
    minim = min(p-1,q-1)
    maxim = max(p-1,q-1)

    base_matrix = np.identity(3)
    base_matrix[minim,maxim] = math.sin(theta)
    base_matrix[maxim,minim] = - math.sin(theta)
    base_matrix[minim,minim] = math.cos(theta)
    base_matrix[maxim,maxim] = math.cos(theta)
    return base_matrix

a = np.arange(9).reshape((3,3))
for i in range(len(theta)):
    rot = rotation_matrix(theta[i], p[i], q[i])
    print(rot)
    print("\n")
    print(a)
    print("\n")
    a = rot @ a

print(a)
