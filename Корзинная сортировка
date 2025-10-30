#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Функция для корзинной сортировки
void bucketSort(vector<int>& arr) {
    if (arr.empty())
        return;

    // Определим минимальный и максимальный элементы массива
    int min_val = *min_element(arr.begin(), arr.end());
    int max_val = *max_element(arr.begin(), arr.end());

    // Определяем число корзин и создаём пустые корзины
    size_t num_buckets = arr.size();
    vector<vector<int>> buckets(num_buckets);

    // Разделяем элементы по корзинам
    for(int value : arr) {
        // Индекс корзины определяется путём нормирования значений
        size_t idx = (value - min_val) * num_buckets / (max_val + 1 - min_val);
        buckets[idx].push_back(value);
    }

    // Внутренняя сортировка каждой корзины
    for(auto &bucket : buckets) {
        sort(bucket.begin(), bucket.end());
    }

    // Заполняем отсортированными элементами исходный массив
    int pos = 0;
    for(const auto &bucket : buckets) {
        for(int value : bucket) {
            arr[pos++] = value;
        }
    }
}

// Основная функция для тестирования
int main() {
    vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    cout << "Исходный массив:" << endl;
    for(int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    bucketSort(arr);

    cout << "Отсортированный массив:" << endl;
    for(int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    return 0;

// Исходный массив: 64 34 25 12 22 11 90 
// Отсортированный массив: 11 12 22 25 34 64 90

}
