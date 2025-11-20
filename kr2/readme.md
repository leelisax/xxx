Задание 13. Генетический алгоритм: задача коммивояжёра
Условие. Найти приближённое решение TSP с помощью генетического алгоритма.
Алгоритм: ГА с упорядоченным кроссовером (OX) и мутацией перестановки.
Язык примера: C++

vector<int> gaTsp(const vector<vector<int>>& dist, int popSize, int generations) {
 vector<vector<int>> population = initPopulation(popSize, dist.size());
 for (int gen = 0; gen < generations; gen++) {
 vector<double> fitness = calcFitness(population, dist);
 // ДОПИСАТЬ: отбор родителей, кроссовер, мутация
 // обновить population
 // Вернуть лучший маршрут
}
