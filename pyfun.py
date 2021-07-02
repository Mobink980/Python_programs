import numpy as np  # python library for numerical calculations
n = int(input())
myList = [n,0]*n
print(myList[0:n],"\n")
a = np.arange(n, n + n**2)
print(a.reshape(n, n),"\n","\n" , np.tril(a.reshape(n, n)))
y=np.ones((n,n),int,n)*n
y[1:-1,1:-1] = 0
print("\n",y, "\n")
z = np.matmul(np.tril(a.reshape(n, n)),y)
print(z,"\n")
z[z >= n**3] *= -1
print(z,"\n","\n",z.max() + z.min())












#a = np.arange(n)
#print(a[::1])
#a = np.reshape(np.arange(9),(3,3))
#c = a[3:8].copy()