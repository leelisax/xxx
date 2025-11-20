// СОРТИРОВКА ВСТАВКАМИ

#include <iostream>
using namespace std;

// Функция для сортировки вставками
void insertionSort(int arr[], int size) {
    for (int i = 1; i < size; ++i) {                // Начинаем с индекса 1
        int key = arr[i];                           // Текущий элемент
        int j = i - 1;                              // Индикатор предыдущих элементов

        // Двигаясь назад, ищем место для вставки текущего элемента
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];                   // Передвигаем предыдущий элемент дальше
            j--;
        }

        // Вставляем текущий элемент на пустое место
        arr[j + 1] = key;
    }
}

// Функция для вывода массива
void printArray(int arr[], int size) {
    for (int i = 0; i < size; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main() {
    int arr[] = {8, 3, 5, 1, 9};                   // Несортированный массив
    int size = sizeof(arr) / sizeof(arr[0]);       // Определяем размер массива

    insertionSort(arr, size);                      // Выполняем сортировку
    cout << "Отсортированный массив: ";
    printArray(arr, size);                         // Выводим отсортированный массив

    return 0;

// Отсортированный массив: 1 3 5 8 9 
    
}
