def funct(x):
    y=0
    y=x*x*x - 5*x - 9
    return y;

def BisectionMethod(x0,x1,epsilon):
    n = 0
    if (funct(x0) * funct(x1) < 0):
        while True:
            
            n += 1;
            x2 = (x0+x1)/2.0
            
            if funct(x0) * funct(x2) < 0:
                x1 = x2
            else:
                x0 = x2
                
                
            if(abs((x0-x1)/x0) < epsilon) : 
                break;
                
                print('The Root is', x0);
                print('Number of iterations =', n);
            else:
                print('Roots do not exist in this interval');
         
x0 = 2;
x1 = 3;
epsilon = 0.0001;
BisectionMethod(x0, x1, epsilon)

        
