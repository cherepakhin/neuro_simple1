import numpy
import matplotlib.pyplot as plt

print('start')

a=numpy.zeros([3,2])
print(a)

a[0,0]=1
a[0,1]=3
a[2,1]=12

print(a)
im=plt.imshow(a, interpolation="nearest")
plt.show()