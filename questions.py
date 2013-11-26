#Alassane Ndiaye
#David Krueger
#Thomas Rohee

import time
import utilitaires
import pylab
import numpy as np
import kernels
from models import SVMLinear, SVMKernel
from preprocesses import TwoMoons, IrisSep, IrisNotSep

grid_size = 50
irisSep = IrisSep.IrisSep()
irisNotSep = IrisNotSep.IrisNotSep()
twoMoons = TwoMoons.TwoMoons()


def polynome(x, p=2):
    new_elements = np.zeros((x.shape[0]))
    for i in range(p):
        poly = np.zeros((x.shape[0]))
        if i == 0:
            poly = np.column_stack((poly, np.ones((poly.shape[0], 1))))
        else:
            k0 = i + 1
            k1 = 0
            while k0 > -1:
                poly = np.column_stack((poly, x[:, 0] ** k0 * x[:, 1] ** k1))
                k0 -= 1
                k1 += 1
        new_elements = np.column_stack((new_elements, poly))

    new_x = np.concatenate((x, new_elements), axis=1)
    return new_x


def results(model, dataset, epochs, epochs_to_display, display_errors, poly=[], kernel_function=None):
    if poly:
        for k in poly:
            print "Polynomial: ", k
            # Polynomial transformation
            transformed_train_set = np.concatenate(
                (polynome(dataset.train_set[:, :-1], k), dataset.train_set[:, -1][:, None]), axis=1)
            transformed_test_inputs = polynome(dataset.test_inputs, k)

            model.train(transformed_train_set, epochs, epochs_to_display, display_errors)

            # Obtenir les sorties sur l'ensemble de test.
            t1 = time.clock()
            les_sorties = model.compute_predictions(transformed_test_inputs)
            t2 = time.clock()
            print 'Took ', t2-t1, ' sec to predict on ', dataset.test_inputs.shape[0], ' points'

            # Convertir les sorties en classe. On prend le signe.
            classes_pred = np.sign(les_sorties)

            # Mesurer la performance.
            err = 1.0 - np.mean(dataset.test_labels == classes_pred)
            print "Test error: ", 100.0 * err, "%"

            # Affichage graphique
            if len(dataset.train_cols) == 2:

                # Surface de decision
                t1 = time.clock()
                les_sorties = model.compute_predictions(transformed_train_set[:, :-1])
                # Convertir les sorties en classe. On prend le signe.
                train_classes_pred = np.sign(les_sorties)
                pylab.scatter(dataset.train_set[:, 0], dataset.train_set[:, 1], c=train_classes_pred)
                pylab.scatter(dataset.test_set[:, 0], dataset.test_set[:, 1], c=classes_pred)
                pylab.show()

                t2 = time.clock()
                print 'Took ', t2-t1, ' sec to predict'
            else:
                print 'Trop de dimensions (', len(dataset.train_cols), ') pour pouvoir afficher la surface de decision'
    else:
        model.train(dataset.train_set, epochs, epochs_to_display, display_errors)

        # Obtenir les sorties sur l'ensemble de test.
        t1 = time.clock()
        les_sorties = model.compute_predictions(dataset.test_inputs)
        t2 = time.clock()
        print 'Took ', t2 - t1, ' sec to predict on ', dataset.test_inputs.shape[0], ' points'

        # Convertir les sorties en classe. On prend le signe.
        classes_pred = np.sign(les_sorties)

        # Mesurer la performance.
        err = 1.0 - np.mean(dataset.test_labels == classes_pred)
        print "Test error: ", 100.0 * err, "%"

        # Affichage graphique
        if len(dataset.train_cols) == 2:
            # Surface de decision
            t1 = time.clock()
            utilitaires.gridplot(model, dataset.train_set, dataset.test_set, n_points=grid_size, select_method=np.sign)
            t2 = time.clock()
            print 'Took ', t2 - t1, ' sec to predict '

        else:
            print 'Trop de dimensions (', len(dataset.train_cols), ') pour pouvoir afficher la surface de decision'


def question21():
    # iris separable
    # Hyper-parameters
    mu = 0.1
    cs = [10 ** 10]
    epochs = 150
    epochs_to_display = [10, 40, 100]
    display_errors = True
    batch = irisSep.train_set.shape[0]

    for c in cs:
        print "Iris separable, C: ", c
        model = SVMLinear.SVMLinear(mu, c, batch)
        results(model, irisSep, epochs, epochs_to_display, display_errors)

    # iris not separable
    # Hyper-parameters
    cs = [0.01, 10, 10 ** 10]
    epochs = 300
    epochs_to_display = []
    display_errors = False
    batch = irisNotSep.train_set.shape[0]

    for c in cs:
        print "Iris not separable, C: ", c
        model = SVMLinear.SVMLinear(mu, c, batch)
        results(model, irisNotSep, epochs, epochs_to_display, display_errors)

    # 2moons
    # Hyper-parameters
    cs = [0.0005, 0.001, 0.1, 1, 10 ** 10]
    epochs = 10000
    batch = twoMoons.train_set.shape[0]

    for c in cs:
        print "2moons, C: ", c
        model = SVMLinear.SVMLinear(mu, c, batch)
        results(model, twoMoons, epochs, epochs_to_display, display_errors)


def question22():
    # 2moons
    # Hyper-parameters
    mu = 0.1
    cs = [0.0005, 0.001, 0.1, 10, 10 ** 3]
    poly = [2, 3, 4]
    epochs = 10000
    epochs_to_display = []
    display_errors = False
    batch = twoMoons.train_set.shape[0]

    for c in cs:
        print "2moons, C: ", c
        model = SVMLinear.SVMLinear(mu, c, batch)
        results(model, twoMoons, epochs, epochs_to_display, display_errors, poly)


def question23():
    # 2moons
    # Hyper-parameters
    mu = 0.1
    cs = [0.001, 0.1, 10, 10 ** 10]
    epochs = 30000
    epochs_to_display = []
    display_errors = False
    kernel_function = kernels.scalar_kernel
    batch = twoMoons.train_set.shape[0]

    for c in cs:
        print "Dot product, C: ", c
        model = SVMKernel.SVMKernel(mu, c, batch, kernel_function)
        results(model, twoMoons, epochs, epochs_to_display, display_errors)

    cs = [0.001, 0.1, 10 ** 5, 10 ** 10]
    poly = [2, 3, 4, 5, 6]
    kernel_function = kernels.polynomial_kernel

    for c in cs:
        for k in poly:
            print "Polynomial, C: " + str(c) + " , k: " + str(k)
            model = SVMKernel.SVMKernel(mu, c, batch, kernel_function, k=k)
            results(model, twoMoons, epochs, epochs_to_display, display_errors)

    cs = [0.001, 10 ** 10]
    sigmas = [0.3, 0.4, 0.5]
    kernel_function = kernels.rbf_kernel

    for c in cs:
        for s in sigmas:
            print "RBF, C: " + str(c) + " , sigma: " + str(s)
            model = SVMKernel.SVMKernel(mu, c, batch, kernel_function, sigma=s)
            results(model, twoMoons, epochs, epochs_to_display, display_errors)