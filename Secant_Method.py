def funct(x):
    y=0;
    y=x*x*x + x-1;
    return y;

def SecantMethod(x0,x1,epsilon):
    n=0;
    
    while abs((x0-x1)/x0) > epsilon:
        x2 = x1 - funct(x0)* (x1-x0)/(funct(x1)-funct(x0));
        x0 = x1;
        x1 = x2;
        
        n += 1
        
    print('The Root is',x0)
    print('Number of Iterations =',n);
    
x0 = -10
x1 = 10 
epsilon = 0.0001
SecantMethod(x0, x1, epsilon);

    