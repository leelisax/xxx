#include <iostream>
#include<list>
using namespace std;

void printL(list<int>& l) {
    for (auto i : l)
        cout << i << " ";
    cout << '\n';
}

int main() {
    
    // создаем пустой список
    list<int> l1;

    // сощздаем список из списка инициализаторов
    stack<int> l2 = {1, 3, 4, 2, 5};

    //создаем список с заданным размером
    stack<int> l3(5, 9);
    
    printL(l1);
    printL(l2);
    printL(l3);
    return 0;
}
