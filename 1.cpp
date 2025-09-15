#include <iostream>
#include <stack>
using namespace std;

// Функция для вывода содержимого стека
void printStack(stack<int> s) {
    while (!s.empty()) {
        cout << s.top() << " ";  // Выводим верхний элемент
        s.pop();                 // Удаляем верхний элемент
    }
    cout << endl;
}

int main() {
    // Создаем пустой стек
    stack<int> s1;
    
    // Создаем стек и заполняем его элементами
    stack<int> s2;
    s2.push(1);  // Добавляем элементы в стек
    s2.push(3);
    s2.push(4);
    s2.push(2);
    s2.push(5);
    
    // Создаем стек с пятью одинаковыми элементами
    stack<int> s3;
    for (int i = 0; i < 5; i++) {
        s3.push(9);  // Добавляем число 9 пять раз
    }
    
    // Вывод стеков
    cout << "Стек 1: ";
    printStack(s1);
    
    cout << "Стек 2: ";
    printStack(s2);
    
    cout << "Стек 3: ";
    printStack(s3);
    
    return 0;
}
