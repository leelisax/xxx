Задание 13. Генетический алгоритм: задача коммивояжёра
- Условие. Найти приближённое решение TSP с помощью генетического алгоритма.
- Алгоритм: ГА с упорядоченным кроссовером (OX) и мутацией перестановки.
- Язык примера: C++

```
vector<int> gaTsp(const vector<vector<int>>& dist, int popSize, int generations) {
 vector<vector<int>> population = initPopulation(popSize, dist.size());
 for (int gen = 0; gen < generations; gen++) {
 vector<double> fitness = calcFitness(population, dist);
 // ДОПИСАТЬ: отбор родителей, кроссовер, мутация
 // обновить population
 // Вернуть лучший маршрут
}
```
- Что дописать: функции отбора (например, турнирным методом), OX‑кроссовера и
мутации перестановки.

### Пошаговое описание алгоритма
### Шаг 1. Инициализация популяции
```
cpp
vector<vector<int>> initPopulation(int popSize, int nCities)
Создаётся popSize случайных перестановок городов 0..nCities‑1.
```
Каждая перестановка — возможный маршрут.
### Шаг 2. Основной цикл по поколениям
Для каждого поколения от 0 до generations ‑ 1:
## а) Вычисление фитнес‑функции
```
cpp
vector<double> calcFitness(const vector<vector<int>>& population, const vector<vector<int>>& dist)
Для каждого маршрута суммируется длина пути:
totalDist = dist[route[0]][route[1]] + dist[route[1]][route[2]] + ... + dist[route[n-1]][route[0]].
```
Фитнес = 1.0 / totalDist (чем короче путь, тем выше фитнес).

Пример:
Маршрут [0, 2, 1, 3] → длина = 100 → фитнес = 0.01.

## б) Создание нового поколения
Для каждой из popSize новых особей:
- Отбор родителей (турнирный, k = 3)
```
cpp
vector<int> selectParent(const vector<vector<int>>& population, const vector<double>& fitness)
```
- Случайным образом выбираются 3 особи из текущей популяции.
- Выбирается особь с максимальным фитнесом.
- Повторяется для второго родителя.
# Кроссовер (Ordered Crossover, OX)
```
cpp
vector<int> orderedCrossover(const vector<int>& parent1, const vector<int>& parent2)
```
- Выбирается случайный сегмент генов (например, позиции 1…2).
- Сегмент из parent1 копируется в потомка.
- Остальные города добавляются из parent2 в порядке их появления, пропуская уже добавленные.
# Мутация
```
cpp
void mutate(vector<int>& individual, double mutationRate = 0.1)
С вероятностью mutationRate (10 %) меняются местами два случайных города.
```
## в) Обновление популяции
Новая популяция заменяет старую.

### Шаг 3. Выбор лучшего решения
- После всех поколений вычисляется фитнес для всех особей финальной популяции.
- Выбирается маршрут с максимальным фитнесом (кратчайшим путём).
- Возвращается этот маршрут.


