#include <stdio.h>

int arr[1000000];

int fibo(int n)
{
    if (arr[n] != -1){
        return arr[n];
    }
    int result = (fibo(n-1) + fibo(n-2))%15746;
    arr[n] = result;
    return result;
}

int main(void)
{
    int n, i;
    for(i=0; i<1000000; i++){
        arr[i] = -1;
    }
    arr[0] = 1;
    arr[1] = 1;

    scanf("%d", &n);
    printf("%d", fibo(n));
}