// ИНТЕРПОЛИРУЮЩИЙ ПОИСК

#include <iostream>
#include <cmath>
using namespace std;

// Функция интерполирующего поиска
int interpolationSearch(int arr[], int size, int x) {
    // Нижняя граница массива
    int low = 0;
    // Верхняя граница массива
    int high = size - 1;
    
    while (low <= high && x >= arr[low] && x <= arr[high]) {
        if (low == high) {   // Если остался единственный элемент
            if (arr[low] == x) return low;
            return -1;
        }
        
        // Вычисление приблизительной позиции
        double pos = low + ((double)(high - low) / (arr[high] - arr[low])) * (x - arr[low]);
        
        // Проверка целочисленности результата
        int index = static_cast<int>(round(pos));
        
        // Элемент найден?
        if (arr[index] == x) return index;
        
        // Продолжаем искать слева или справа
        else if (arr[index] > x) high = index - 1;
        else low = index + 1;
    }
    
    return -1; // Элемент не найден
}

int main() {
    int arr[] = {10, 12, 13, 16, 18, 19, 20, 21};
    int size = sizeof(arr)/sizeof(arr[0]);
    int key = 18;
    
    int result = interpolationSearch(arr, size, key);
    
    if(result != -1) cout << "Element found at position " << result << endl;
    else cout << "Element not found in the array." << endl;
    
    return 0;

// Element found at position 4

}
