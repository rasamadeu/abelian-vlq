import numpy as np

def get_texture_zero_from_matrix(matrix):
    
	texture = np.array([], dtype = "uint8")
	collumn, line = np.shape(matrix)

	for i in range(collumn):
		for j in range(line):
			if matrix[i, j] == 0:
				texture = np.append(texture, j + i*line)

	return  texture, collumn, line

def get_matrix_from_texture_zero(texture, n, m = 0):
	
	if m == 0:
		m = n

	matrix = np.ones([n, m], dtype = "uint8")
	
	for i in np.nditer(texture):
		matrix[int(i/m),int(i%m)] = 0

	return matrix

def get_permutation_from_matrix(matrix):
	
	permutation = np.array([], dtype = "uint8")
	n, _ = np.shape(matrix)

	for i in range(n):
		for j in range(n):
			if matrix[i, j] == 1:
				permutation = np.append(permutation, j)

	return  permutation

def texture_permutation__mult(texture, line, permutation):

	result = np.array([], dtype = "uint8")
	n_perm = np.size(permutation)
	n_text = np.size(texture)

	for i in range(n_perm):
		for j in range(n_text):
			if texture[j] % line == permutation[i]:
				result = np.append(result, texture[j] + (i-permutation[i]))

	return np.sort(result)
