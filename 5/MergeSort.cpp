// СОРТИРОВКА СЛИЯНИЕМ

#include <iostream>
using namespace std;

// Функция для объединения двух отсортированных половин массива
void merge(int arr[], int left, int mid, int right) {
    // Определяем размеры двух подпоследовательностей
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Создаем временные массивы
    int L[n1], R[n2];

    // Копируем данные в временные массивы
    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    // Индексы начальных элементов первой и второй половин
    int i = 0, j = 0;
    int k = left; // Индекс начальной точки в исходном массиве

    // Объединяем две половины
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) { // Берём меньший элемент
            arr[k++] = L[i++];
        }
        else {
            arr[k++] = R[j++];
        }
    }

    // Заполняем оставшиеся элементы левой половины
    while (i < n1) {
        arr[k++] = L[i++];
    }

    // Заполняем оставшиеся элементы правой половины
    while (j < n2) {
        arr[k++] = R[j++];
    }
}

// Рекурсивная функция сортировки
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        // Находим середину
        int mid = left + (right - left) / 2;

        // Сортируем левую половину
        mergeSort(arr, left, mid);

        // Сортируем правую половину
        mergeSort(arr, mid + 1, right);

        // Объединяем отсортированные половины
        merge(arr, left, mid, right);
    }
}

// Вспомогательная функция для вывода массива
void printArray(int A[], int size) {
    for (int i = 0; i < size; i++)
        cout << A[i] << " ";
    cout << endl;
}

// Главная функция
int main() {
    int arr[] = {12, 11, 13, 5, 6};
    int arr_size = sizeof(arr)/sizeof(arr[0]);

    cout << "Исходный массив:";
    printArray(arr, arr_size);

    mergeSort(arr, 0, arr_size - 1); // Выполняем сортировку

    cout << "Отсортированный массив:";
    printArray(arr, arr_size);

    return 0;


// Исходный массив:12 11 13 5 6 
// Отсортированный массив:5 6 11 12 13 
  
}
