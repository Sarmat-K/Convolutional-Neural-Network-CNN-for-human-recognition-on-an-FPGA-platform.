matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Левый треугольник
left_triangle = []
for i in range(len(matrix)):
    if i < len(matrix)/2:
        for j in range(i+1):
            left_triangle.append(matrix[i][j])
    else:
        for j in range(len(matrix)-i):
            left_triangle.append(matrix[i][j])

# Правый треугольник
right_triangle = []
for i in range(len(matrix)-1, -1 , -1):
    if i < len(matrix)//2:
        for j in range(len(matrix)-1,len(matrix)-2-i,-1):
            right_triangle.append(matrix[i][j])
    else:
        for j in range(len(matrix)-1,i-1,-1):
            right_triangle.append(matrix[i][j])



# Вывод результатов
print("Левый треугольник:", left_triangle)
print("Правый треугольник:", right_triangle)
