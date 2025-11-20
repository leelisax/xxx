// БИНАРНЫЙ ПОИСК

#include <iostream>
using namespace std;

// Функция бинарного поиска
int binarySearch(int arr[], int target, int left, int right) {
    while (left <= right) {
        // Вычисление среднего индекса
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) { // Нашли элемент!
            return mid;
        }
        else if (arr[mid] > target) { // Элемент расположен слева от центра
            right = mid - 1;
        }
        else { // Элемент расположен справа от центра
            left = mid + 1;
        }
    }
    
    // Элемента нет в массиве
    return -1;
}

int main() {
    int arr[] = {1, 3, 5, 7, 9, 11}; // Отсортированный массив
    int size = sizeof(arr)/sizeof(arr[0]);
    int target = 7;                  // Искусственный пример

    int result = binarySearch(arr, target, 0, size - 1);

    if(result != -1)
        cout << "Элемент найден на позиции " << result << endl;
    else
        cout << "Элемент не найден." << endl;

    return 0;

// Элемент найден на позиции 3
  
}
