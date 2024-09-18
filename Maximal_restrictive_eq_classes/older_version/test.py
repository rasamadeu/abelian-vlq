import texture_zeros as tz
import numpy as np
import time

def main():

    start = time.time()

    a = np.array([[1,0,0,1], [1,0,1,1], [0,0,0,1], [0,1,1,0]])
    b = np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]])
    c = np.array([[0,1,0,0], [0,0,1,0], [1,0,0,0], [0,0,0,1]])

    #for i in range(1000000):
     #   a_text
      #  b&a&c

    print(tz.get_permutation_from_matrix(c))
    end = time.time()

    print("Runtime", end-start)
    b2,_,_ = get_texture_zero_from_matrix(b)
    c2,_,_ = get_permutation_from_matrix(c)
    print(get_matrix_from_texture_zero(texture_permutation_mult(b2,c2)))

if __name__ == "__main__":
    main()
