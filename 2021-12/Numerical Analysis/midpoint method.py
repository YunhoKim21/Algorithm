import math

n=10; a=0; b=1; alpha = 1
h = (b-a)/n
w = alpha

def derivative(t, y):
    return y

t = a

print('f = exp(x), a = {}, b = {}, n = {}, alpha = {}\n'.format(a, b, n, alpha))

print('{:>5} {:>10} {:>10}'.format('t', 'midpoint', 'ans'))
for i in range(n+1):
    t = a + h*i
    print('{:.3f} {:10f} {:10f}'.format(t, w, math.exp(t)))
    w += h * derivative(t + 1/2 * h, w + 1/2 * h * derivative(t, w))