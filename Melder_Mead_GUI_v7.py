import copy
import csv

selection = 'Rosenbrock'

'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]
        best_res = res

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print('...best so far:', best, '...res:', res[0][0][1])
        row1 = [res[0][0][0], res[0][0][1], res[0][1], str(iters), selection]
        row2 = [res[1][0][0], res[1][0][1], res[1][1], str(iters), selection]
        row3 = [res[2][0][0], res[2][0][1], res[2][1], str(iters), selection]
        row4 = [res[3][0][0], res[3][0][1], res[3][1], str(iters), selection]

        with open('nelder_mead.csv', 'a') as cs:
            write = csv.writer(cs)
            write.writerow(row1)
            write.writerow(row2)
            write.writerow(row3)
            write.writerow(row4)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


# Optimization Function
def f(v):
    if selection == "Himmelblau":
        z = (v[0]**2 + v[1] - 11)**2 + (v[0] + v[1]**2 - 7)**2  # Himmelblau-Funktion
    else:
        z = (1 - v[0])**2 + 100 * ((v[1] - (v[0]**2))**2)  # Rosenbrock-Funktion

    return z


if __name__ == "__main__":
    # test
    import numpy as np

    header = ["x1", "x2", "f(x1, x2)", "Iteration", "Algorithmus"]  # header of the csv-sheet
    with open('nelder_mead.csv', 'w') as cs:
        write = csv.writer(cs)
        write.writerow(header)

    print(nelder_mead(f, np.array([6., 9., 30.])))
