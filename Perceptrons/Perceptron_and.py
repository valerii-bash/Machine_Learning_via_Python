from numpy import array
from functools import reduce

class Perceptron_and:
    """
        Перцептрон, способный находить значение логической функции "И" от n аргументов.

        Содержит методы:

        __create_all_numbers(self, n)

        __create_correct_answers(self, all_numbers)

        __create_all_weights(self, n)

        __sum_function(self, inputs)

        __activation_function(self, inputs)

        __train_on_single_example(self, example, correct_answer, learning_rate)

        learning_function(self, n, max_steps, learning_rate)

        result(self, inputs)
    """

    def __create_all_numbers(self, n):
        """
        Функция-генератор!
        Метод принимает целое число n(число аргументов)
        И возвращает вектор всех чисел от 0 до n в двоичной системе счисления (2**n чисел),

        Example input:
        2
        Example output:
        [[0 0]
         [0 1]
         [1 0]
         [1 1]]
        """
        for number in ([ int(item) for item in bin(i)[2:]] for i in range(2**n)):
         if len(number) < n:
            yield array([0]*(n - len(number))+number)
         else:
             yield array(number)

    def __create_correct_answers(self, all_numbers):
        """
        Функция-генератор!
        Метод принимает вектор входных данных
        И возвращает вектор результатов от логической функции "И"

        Example input:
        [[0 0]
         [0 1]
         [1 0]
         [1 1]]
        Example output:
        [[0]
         [0]
         [0]
         [1]]
        """
        for number in all_numbers:
            yield array(reduce(lambda x, y: x and y, number))

    def __create_all_weights(self, n):
        """
        Метод принимает целое число n
        И возвращает ГОРИЗОНТАЛЬНЫЙ вектор весов ( включая смещение, которое является 1м элементом вектора)

        Example input:
        3
        Example output:
        [0 0 0]
        """
        self.__weights = array([0] * n)

    def __sum_function(self, inputs):
        """
        Сумматорная функция.
        Принимает входной вектор inputs.
        И возвращает матричное произведение вектора весов на вектор входов."""

        return self.__weights.dot(inputs)

    def __activation_function(self, inputs):
        """
        Активационная функция.
        Возращает 1, если результат сумматорной функции больше или равен 1
        И 0 иначе."""

        return 1 if self.__sum_function(inputs) >= 1 else 0

    def __train_on_single_example(self, example, correct_answer, learning_rate):
        """
        Метод, изменяющий значения весов на основании вектора входов example.
        Возвращает ошибку."""

        predicted_answer = self.__activation_function(example)
        error = int(correct_answer - predicted_answer)
        self.__weights = self.__weights + learning_rate * example.dot(error)

        return error

    def learning_function(self, n, max_steps, learning_rate):
        """
        Обучающая функция.
        Принимает на вход:
                    n - кол-во аргументов функции
                    max_steps - максимальное кол-во шагов
                    learning_rate - скорость обучения"""

        self.__create_all_weights(n)

        current_step = 0
        errors = 1

        while errors and current_step < max_steps:
            current_step += 1
            errors = 0
            for example, answer in zip(self.__create_all_numbers(n),self.__create_correct_answers(self.__create_all_numbers(n))):
                error = self.__train_on_single_example(example,answer, learning_rate)
                errors += int(error)

    def result(self, inputs):
        """
        Метод принимает на вход пример
        И возвращает ответ перцетрона.

        Example input:
         [0, 0]
         [0, 1]
         [1, 0]
         [1, 1]
        Example output:
         0
         0
         0
         1"""

        perceptron_inputs = inputs[:]
        return self.__activation_function(perceptron_inputs)