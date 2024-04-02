#Задание 4 часть 1.
#С использованием numpy:
#1) создать несколько (10-20) списков чисел с плавающей точкой и заполнить их случайными числами от 0 до 1 с точностью до 4го знака
#2)методом append() собрать из этих списков матрицу

#Задание 4 часть 2.
#С использованием tensorflow:
#1) собрать тензор ранга 3 из матриц с первой части задания
#2) провести над тензором 3-4 математических операции
#3) сделать 2 среза из тензора по всем осям и объединить их новый тензор

import numpy as np
import tensorflow as tf

def MatrixCreate():
    lists = []
    for i in range(10):
        random_list = np.array(np.random.rand(6), dtype=np.float_).round(4)
        lists.append(random_list)
    matrix = np.array(lists)
    return matrix


Matrix1 = MatrixCreate()
print(Matrix1, '\n')
Matrix2 = MatrixCreate()
print(Matrix2, '\n')
Matrix3 = MatrixCreate()
print(Matrix3, '\n')

Matrix4 = MatrixCreate()
print(Matrix4, '\n')
Matrix5 = MatrixCreate()
print(Matrix5, '\n')
Matrix6 = MatrixCreate()
print(Matrix6, '\n')


rank_3_tensor = tf.constant([Matrix1, Matrix2, Matrix3])
print('Первый тензор: \n', rank_3_tensor, '\n')

another_rank_3_tensor = tf.constant([Matrix4, Matrix5, Matrix6])
print('Второй тензор: \n', another_rank_3_tensor, '\n')

sum = rank_3_tensor + another_rank_3_tensor
print('Сумма тензоров: \n', sum, '\n')

difference = rank_3_tensor - another_rank_3_tensor
print('Разность тензоров: \n', difference, '\n')

mult = tf.multiply(rank_3_tensor, another_rank_3_tensor)
print('Поэлементное умножение тензоров: \n', mult, '\n')



slice1 = rank_3_tensor[1:3, 0:2, 0:3]
slice2 = another_rank_3_tensor[1:3, 0:2, 0:3]

print('С1:', slice1, '\n')
print('C2', slice2, '\n')
new_tensor = tf.concat([slice1, slice2], axis=0)
print("Новый тензор после объединения срезов:")
print(new_tensor)