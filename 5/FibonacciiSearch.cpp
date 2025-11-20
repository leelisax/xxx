/ ПОИСК ФИБОНАЧЧИ

#include <iostream>
using namespace std;

// Функция поиска элемента методом Фибоначчи
int fibonacciSearch(int arr[], int x, int n) {
    // Первые три числа Фибоначчи
    int fibMm2 = 0;   // F(m-2)
    int fibMm1 = 1;   // F(m-1)
    int fibM = fibMm2 + fibMm1; // F(m)
    
    // Находим наименьшее число Фибоначчи большее размера массива
    while (fibM < n) {
        fibMm2 = fibMm1;
        fibMm1 = fibM;
        fibM = fibMm2 + fibMm1;
    }
    
    // Сохраняем индекс начала окна поиска
    int offset = -1;
    
    // Пока размер окна больше единицы
    while (fibM > 1) {
        // Определяем минимальный возможный индекс внутри текущего окна
        int i = min(offset + fibMm2, n - 1);
        
        // Если элемент меньше искомого, уменьшаем окно справа
        if (arr[i] < x) {
            fibM = fibMm1;
            fibMm1 = fibMm2;
            fibMm2 = fibM - fibMm1;
            offset = i;
        }
        // Если элемент больше искомого, уменьшаем окно слева
        else if (arr[i] > x) {
            fibM = fibMm2;
            fibMm1 -= fibMm2;
            fibMm2 = fibM - fibMm1;
        }
        // Элемент найден
        else {
            return i;
        }
    }
    
    // Проверяем последний оставшийся элемент
    if (fibMm1 && offset+1<n && arr[offset+1]==x){
        return offset+1;
    }
    
    // Элемент не найден
    return -1;
}

int main() {
    int arr[] = {10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100};
    int x = 85;
    int n = sizeof(arr)/sizeof(arr[0]);
    
    int result = fibonacciSearch(arr, x, n);
    
    if(result != -1) {
        cout << "Element found at index: " << result << endl;
    } else {
        cout << "Element not found in array." << endl;
    }
    
    return 0;
    
    // Element found at index: 8
    
}
