
import sympy as sy

def legendre(N):
    
    x,y = sy.symbols('x,y')
    # number of coefficients to use in approximation
    
    # create the coefficients a
    a = []
    for j in range(0,N):
       a.append(sy.sympify('a'+str(j)))
    
    # create the legendre polynomials
    lx = []
    lx.append(sy.sympify(1))
    lx.append(sy.sympify(x))
    for n in range(1,N):
       lx.append(sy.simplify((2*n+1)*x*lx[n]/(n+1)-(n/(n+1))*lx[n-1]))
    
    return lx


def alternate(s):
    check=True
    visited = set()
    length = 0
    for j in s:
        for k in s:
            if j != k and (j,k) not in visited and (k,j) not in visited:
                visited.add((j,k))
                visited.add((k,j))
                shortlist = [l for l in s if l==j or l==k]
                
                check=True
                for i in range(len(shortlist)-1):
                    if shortlist[i]==shortlist[i+1]:
                        check=False
                        break
                if check:
                    if len(shortlist) > length:
                        length = len(shortlist)  
                    
    return length

def andProduct(a, b):
    
    x = bin(a)[2:]
    y = bin(b)[2:]
    if len(x) != len(y):
        x = str(0)* (len(y)-len(x)) + x
    answer = ''
    n = len(x)
    for i in range(n):
        if x[i] == y[i]:
            answer += x[i]
        else:
            answer += str(0)* (n-i)
            break
            
    
    return int(answer,2)
        