import math

n=10; a=1; b=2; alpha = 0
h = (b-a)/n
w = alpha

def derivative(t, y):
    return 2 * (y/t) + t ** 2 * math.exp(t)

for i in range(n):
    t = a + h * i
    print ('{:.5f} {:.10f} {:.10f}'.format(t, w, t * t *(math.exp(t) - math.exp(1))))
    w += h * derivative(t, w)