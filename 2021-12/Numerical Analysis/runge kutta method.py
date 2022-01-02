import math

n=10; a=0; b=1; alpha = 1
h = (b-a)/n
w = alpha

def derivative(t, y):
    return y

t = a

print('f = exp(x), a = {}, b = {}, n = {}, alpha = {}\n'.format(a, b, n, alpha))

print('{:>5} {:>10} {:>10}'.format('t', 'rk4', 'ans'))
for i in range(n+1):
    t = a + h*i
    print('{:.3f} {:10f} {:10f}'.format(t, w, math.exp(t)))
    k1 = h * derivative(t, w)
    k2 = h * derivative(t + h/2, w + k1/2)
    k3 = h * derivative(t + h/2, w + k2/2)
    k4 = h * derivative(t + h, w + k3)
    w += (k1 + 2 * k2 + 2 * k3 + k4)/6