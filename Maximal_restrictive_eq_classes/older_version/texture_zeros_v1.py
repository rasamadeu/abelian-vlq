import numpy as np


def texture_zeros(
        dim, n_zeros,
        n_control=0, texture_rec=np.array([]), i_rec=0, j_rec=0):

    if n_control == 0:

        if n_zeros == 1:

            texture = np.empty(
                (dim**2, dim, dim), dtype="uint8")

            for i in range(dim):
                for j in range(dim):
                    texture_iter = np.ones([dim, dim])
                    texture_iter[i, j] = 0
                    texture[i * dim + j] = texture_iter
            return texture

        texture = np.empty(
            combinations, dim, dim)
        k = 0

        for i in range(dim):
            for j in range(dim):
                texture_iter = np.ones([dim, dim])
                texture_iter[i, j] = 0
                texture_rec = texture_zeros(
                    dim, n_zeros, n_control + 1, texture_iter, i, j)
                for element in texture_rec:
                    texture[k] = element
                    k + +

        return texture

    elif n_control == n_zeros - 1:

        for j_rec in range()

        if j_rec == dim - 1:
            texture_iter[i_rec + 1, 0] = 0

        else:
            texture_iter[i_rec, j_rec + 1] = 0

        return texture_iter

    else:

        if j_rec == dim - 1:
            texture_iter[i_rec + 1, 0] = 0
            return texture_zeros(dim, n_zeros, n_control + 1, texture_iter, i)

        else:
            texture_iter[i_rec, j_rec + 1] = 0

        return 0
