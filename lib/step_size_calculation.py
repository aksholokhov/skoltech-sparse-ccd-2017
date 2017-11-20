from scipy import sparse
import numpy as np

class BaseStepSizeCalculator(object):

    def get_step_size(self, x, coord):
        raise Exception("Not implemented yet")

class ConstantStepSize(BaseStepSizeCalculator):

    def __init__(self, start_num = 10):
        self.__i = start_num

    def get_step_size(self, x, coord):
        self.__i += 1
        return 1/(1 + self.__i)

class CoordParabolicStepSize(BaseStepSizeCalculator):

    def __init__(self, f, alpha = 0.3, init_func_value = None):
        self.__f = f
        self.__alpha = alpha
        self.__init_func_value = init_func_value
        if init_func_value is not None:
            self.__f_x = init_func_value

    def get_step_size(self, x, coord):
        h = sparse.csr_matrix((1, max(x.shape) + 1))
        h[0, 0] = 1
        h[0, coord] = 1

        if self.__init_func_value is None:
            f_x = self.__f(x)
        else:
            f_x = self.__f_x
        f_x1 = self.__f((1 - self.__alpha) * x + self.__alpha * h)
        f_x2 = self.__f((1 - 2*self.__alpha) * x + 2 * self.__alpha * h)
        gamma = - 0.5*self.__alpha*(4*f_x1 - 3*f_x - f_x2)/(f_x2 - 2*f_x1 + f_x)
        if self.__init_func_value is not None:
            self.__f_x = self.__f((1 - gamma) * x + gamma * h)
        return gamma

class RidgeParabolicStepSize(BaseStepSizeCalculator):

    def __f_move(self, alpha, j):
        xAh = self.__xAh
        A = self.__AT
        mu = self.__mu
        x = self.__x
        yTy = self.__yTy
        fx = self.__f_x


        result = 2 * alpha * (1 - alpha) * xAh
        result += alpha ** 2 * (yTy + 2 * A[j].dot(A[0].T) + A[j].dot(A[j].T))
        return result[0, 0] - mu*alpha + mu/2*alpha**2 + mu/2*(alpha**2*2) + \
               mu/2*(2 * alpha * (1 - alpha) * (1 + x[0, j])) + (1-alpha)**2*fx

    def __init__(self, A, x0, f_x0, mu, alpha = 0.1):
        self.__AT = A.T
        self.__x = x0
        self.__alpha = alpha
        self.__mu = mu
        self.__Ax = A.dot(x0.T).T
        self.__yTy = self.__AT[0].dot(self.__AT[0].T)
        self.__prev_min_coord = None
        self.__prev_gamma = None
        self.__f_x = f_x0

    def get_step_size(self, x, coord):
        if self.__prev_min_coord is not None:
            self.__Ax = (1 - self.__prev_gamma) * self.__Ax + \
                        self.__prev_gamma * (self.__AT[0] + self.__AT[self.__prev_min_coord])

        self.__Ah = self.__AT[coord] + self.__AT[0]
        self.__xAh = self.__Ax.dot(self.__Ah.T)

        f_x1 = self.__f_move(self.__alpha, coord)
        f_x2 = self.__f_move(2 * self.__alpha, coord)

        gamma = - 0.5 * self.__alpha * (4 * f_x1 - 3 * self.__f_x - f_x2) / (f_x2 - 2 * f_x1 + self.__f_x)
        self.__f_x = self.__f_move(gamma, coord)

        if abs(gamma) >= 1:
            gamma = np.sign(gamma) * 0.99

        self.__prev_min_coord = coord
        self.__prev_gamma = gamma
        return gamma