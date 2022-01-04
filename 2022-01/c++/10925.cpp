#include <iostream>
#include <vector>

using namespace std;

int main(void)
{
    vector<int> v;
    while(true)
    {
        int a, b;
        cin >> a;
        cin >> b;
        if (a == 0 && b == 0)
        {
            break;
        }
        v.push_back(a + b);
    }
    vector<int>::iterator itor = v.begin();
    for(; itor!= v.end(); itor ++)
    {
        cout<<*itor<<endl;
    }
}