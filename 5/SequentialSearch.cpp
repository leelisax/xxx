// ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК

#include <iostream>
using namespace std;

// Функция реализации последовательного поиска
int sequentialSearch(int arr[], int size, int target) {
    for (int i = 0; i < size; ++i) { // Просматриваем каждый элемент массива
        if (arr[i] == target) {      // Если найден нужный элемент
            return i;                // Возвращаем индекс найденного элемента
        }
    }
    return -1;                      // Элемент не найден
}

int main() {
    int array[] = {10, 20, 80, 30, 60, 50};
    int n = sizeof(array)/sizeof(array[0]);   // Определяем размер массива
    int x = 80;                              // Искомый элемент

    int result = sequentialSearch(array, n, x);
    
    if(result != -1)
        cout << "Элемент найден на индексе: " << result << endl;
    else
        cout << "Элемент не найден." << endl;
        
    return 0;

// Элемент найден на индексе: 2
  
}
