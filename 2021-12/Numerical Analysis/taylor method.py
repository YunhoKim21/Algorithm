import math

n=10; a=0; b=1; alpha = 1
h = (b-a)/n
w = alpha
w2 = alpha 
w3 = alpha

print('f = exp(x), a = {}, b = {}, n = {}, alpha = {}\n'.format(a, b, n, alpha))
def derivative(t, y):
    return y

def second_derivative(t, y):
    return y

def third_derivative(t, y):
    return y

t=a

print('{:>5} {:>10} {:>10} {:>10} {:>10}'.format('t', 'euler(1)', 'taylor(2)', 'taylor(3)','ans'))
for i in range(n+1):
    t = a + h*i
    print('{:.3f} {:10f} {:10f} {:10f} {:10f}'.format(t, w, w2, w3,math.exp(t)))
    w += h * derivative(t, w)
    w2 += h * derivative(t, w2) + 0.5 * h**2 * second_derivative(t, w2)
    w3 += h * derivative(t, w3) + 0.5 * h**2 * second_derivative(t, w3) + 1/6 * h**3 * third_derivative(t, w3)
