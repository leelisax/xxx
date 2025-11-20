// ПИРАМИДАЛЬНАЯ СОРТИРОВКА

#include <iostream>
using namespace std;

// Функция max_heapify восстанавливает свойство кучи для узла i
void heapify(int arr[], int n, int i) {
    // Предполагаем, что корень является наибольшим элементом
    int largest = i;
    int left_child = 2 * i + 1;   // Левый потомок
    int right_child = 2 * i + 2;  // Правый потомок

    // Если левый ребенок существует и больше корня
    if (left_child < n && arr[left_child] > arr[largest])
        largest = left_child;
    
    // Если правый ребенок существует и больше текущего самого большого
    if (right_child < n && arr[right_child] > arr[largest])
        largest = right_child;

    // Если самый большой элемент изменился, меняем местами
    if (largest != i) {
        swap(arr[i], arr[largest]);
        
        // Рекурсивно выполняем heapify для нового поддерева
        heapify(arr, n, largest);
    }
}

// Основная функция для сортировки массива методом heap sort
void heap_sort(int arr[], int n) {
    // Строим максимальную кучу (max-heap)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);  // Приводим к состоянию кучи начиная снизу вверх

    // Один за другим извлекаем элементы из кучи
    for (int i = n - 1; i > 0; i--) {
        // Перемещаем текущий корневой элемент (самый большой) в конец массива
        swap(arr[0], arr[i]); 
        
        // Перестраиваем кучу для оставшихся элементов
        heapify(arr, i, 0);  // Обрабатываем оставшиеся элементы с новым размером kучи
    }
}

// Тестовая функция для проверки правильности сортировки
void print_array(const int arr[], int size) {
    cout << "Отсортированный массив: ";
    for (int i = 0; i < size; ++i)
        cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(arr)/sizeof(arr[0]);

    heap_sort(arr, n);

    print_array(arr, n);

    return 0;

// Отсортированный массив: 5 6 7 11 12 13

}
