// СОРТИРОВКА ШЕЛЛА

#include <iostream>
using namespace std;

// Функция ShellSort
void shellSort(int arr[], int n) {
    // Начальное значение шага gap берется примерно половиной длины массива
    for (int gap = n / 2; gap > 0; gap /= 2) {   // Уменьшаем шаг вдвое на каждом проходе
        // Проходим по каждому интервалу начиная с первого возможного индекса
        for (int i = gap; i < n; i++) {
            int temp = arr[i];                  // Сохраняем текущий элемент
            
            // Сравниваем и перемещаем элементы на расстояние gap назад,
            // если текущий элемент меньше предыдущего
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;                      // Устанавливаем текущий элемент на правильное место
        }
    }
}

// Функция печати массива
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int arr[] = {12, 34, 54, 2, 3};
    int n = sizeof(arr)/sizeof(arr[0]);
    
    cout << "Исходный массив:\\n";
    printArray(arr, n);
    
    shellSort(arr, n);                          // Сортируем массив методом ShellSort
    
    cout << "\\nОтсортированный массив:\\n";
    printArray(arr, n);
    
    return 0;

// Исходный массив: 12 34 54 2 3 
// Отсортированный массив: 2 3 12 34 54
  
}
