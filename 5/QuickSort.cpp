// БЫСТРАЯ СОРТИРОВКА

#include <iostream>
#include <vector>

// Выбор опорного элемента (pivot) и разделение массива
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];     // Опорный элемент выбран последним элементом
    int i = low - 1;           // Индекс для размещения меньших элементов

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) { // Если текущий элемент меньше или равен опорному
            i++;              // Перемещаем индекс для малых элементов вперед
            std::swap(arr[i], arr[j]); // Меняем местами малый элемент и текущий
        }
    }
    std::swap(arr[i+1], arr[high]); // Ставим опорный элемент на своё место
    return i + 1;                   // Возвращаем позицию опорного элемента
}

// Основная процедура Quicksort
void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);       // Получаем позицию опорного элемента
        
        // РЕКУРСИЯ: сортируем левую и правую части массива
        quickSort(arr, low, pi - 1);             // До опорного элемента
        quickSort(arr, pi + 1, high);            // После опорного элемента
    }
}

int main() {
    std::vector<int> arr = {10, 7, 8, 9, 1, 5}; // Тестовый массив
    int size = arr.size();

    quickSort(arr, 0, size - 1);               // Выполняем быструю сортировку

    std::cout << "Отсортированный массив: ";
    for (int i = 0; i < size; i++)
        std::cout << arr[i] << " ";             // Печать результата
    std::cout << "\\n";

    return 0;

// Отсортированный массив: 1 5 7 8 9 10
  
}
