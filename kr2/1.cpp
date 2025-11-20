#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

using namespace std;

// 1. Инициализация популяции: случайные перестановки городов
vector<vector<int>> initPopulation(int popSize, int nCities) {
    vector<vector<int>> population;
    vector<int> cities(nCities);
    iota(cities.begin(), cities.end(), 0);  // 0, 1, ..., nCities-1

    for (int i = 0; i < popSize; ++i) {
        vector<int> individual = cities;
        shuffle(individual.begin(), individual.end(), mt19937(random_device{}()));
        population.push_back(individual);
    }
    return population;
}

// 2. Вычисление фитнес-функции: обратная величина длины маршрута
vector<double> calcFitness(const vector<vector<int>>& population, const vector<vector<int>>& dist) {
    vector<double> fitness;
    for (const auto& route : population) {
        double totalDist = 0.0;
        int n = route.size();
        for (int i = 0; i < n; ++i) {
            totalDist += dist[route[i]][route[(i + 1) % n]];
        }
        fitness.push_back(1.0 / totalDist);  // Чем короче маршрут, тем выше фитнес
    }
    return fitness;
}

// 3. Турнирный отбор родителя (k = 3)
vector<int> selectParent(const vector<vector<int>>& population, const vector<double>& fitness) {
    int n = population.size();
    const int k = 3;  // Размер турнира
    vector<int> candidates(k);

    // Выбираем k случайных кандидатов
    for (int i = 0; i < k; ++i) {
        candidates[i] = rand() % n;
    }

    // Находим кандидата с максимальным фитнесом
    int bestIdx = candidates[0];
    for (int i = 1; i < k; ++i) {
        if (fitness[candidates[i]] > fitness[bestIdx]) {
            bestIdx = candidates[i];
        }
    }
    return population[bestIdx];
}

// 4. Упорядоченный кроссовер (Ordered Crossover, OX)
vector<int> orderedCrossover(const vector<int>& parent1, const vector<int>& parent2) {
    int n = parent1.size();
    vector<int> child(n, -1);  // -1 означает "пусто"

    // Выбираем случайный отрезок [start, end]
    int start = rand() % n;
    int end = rand() % n;
    if (start > end) swap(start, end);

    // Копируем отрезок из parent1 в child
    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }

    // Заполняем оставшиеся позиции из parent2 (в порядке следования)
    int idx = (end + 1) % n;  // начинаем после конца отрезка
    for (int city : parent2) {
        // Если city уже есть в child, пропускаем
        if (find(child.begin(), child.end(), city) != child.end()) {
            continue;
        }
        // Вставляем city в первую свободную позицию
        while (child[idx] != -1) {
            idx = (idx + 1) % n;
        }
        child[idx] = city;
    }

    return child;
}

// 5. Мутация: случайная перестановка двух городов
void mutate(vector<int>& individual, double mutationRate = 0.1) {
    if ((double)rand() / RAND_MAX < mutationRate) {
        int i = rand() % individual.size();
        int j = rand() % individual.size();
        swap(individual[i], individual[j]);
    }
}

// 6. Основной алгоритм TSP с ГА
vector<int> gaTsp(const vector<vector<int>>& dist, int popSize, int generations) {
    int nCities = dist.size();
    vector<vector<int>> population = initPopulation(popSize, nCities);

    for (int gen = 0; gen < generations; ++gen) {
        vector<double> fitness = calcFitness(population, dist);

        // Создаём новое поколение
        vector<vector<int>> newPopulation;

        for (int i = 0; i < popSize; ++i) {
            // Отбираем двух родителей турнирным методом
            vector<int> parent1 = selectParent(population, fitness);
            vector<int> parent2 = selectParent(population, fitness);

            // Применяем кроссовер
            vector<int> child = orderedCrossover(parent1, parent2);

            // Применяем мутацию
            mutate(child);

            newPopulation.push_back(child);
        }

        population = newPopulation;  // Обновляем популяцию
    }

    // Находим лучший маршрут в финальной популяции
    vector<double> finalFitness = calcFitness(population, dist);
    int bestIdx = 0;
    for (int i = 1; i < popSize; ++i) {
        if (finalFitness[i] > finalFitness[bestIdx]) {
            bestIdx = i;
        }
    }

    return population[bestIdx];  // Возвращаем лучший маршрут
}
