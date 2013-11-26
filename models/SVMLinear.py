#Alassane Ndiaye
#David Krueger
#Thomas Rohee

import pylab
import numpy as np
import utilitaires


class SVMLinear:
    def __init__(self, mu, c_errors, batch):
        self.batch = batch
        self.c_errors = c_errors
        self.mu = mu

    def train(self, train_data, epochs, epochs_to_display=[], display_errors=False):
        self.weights = np.random.random(train_data.shape[1] - 1)
        self.biais = 0
        n_examples = train_data.shape[0]
        data = train_data[:, :-1]
        targets = train_data[:, -1]

        average_loss_array = []
        class_error_rate_array = []
        for j in range(epochs):

            if j + 1 in epochs_to_display:
                print "epoch: ", j + 1
                utilitaires.gridplot(self, train_data, train_data, 50, select_method=np.sign)

            if j % n_examples / self.batch == 0:
                np.random.shuffle(range(n_examples))
            wgrads = 0
            bgrads = 0
            for l in range(self.batch):
                i = (j * self.batch + l) % n_examples

                if (np.dot(self.weights, data[i]) + self.biais) * targets[i] >= 1:
                    wgrads += self.weights / (n_examples ** 2 * self.c_errors)
                else:
                    wgrads += (- data[i] * targets[i] + self.weights / (n_examples * self.c_errors)) / n_examples
                    bgrads = bgrads - targets[i] / n_examples

            self.weights -= self.mu * wgrads
            self.biais -= self.mu * bgrads

            if display_errors:
                class_error_rate = (1 - (np.sign(np.dot(data, self.weights) + self.biais) == targets).mean())
                average_loss = 1 / (2 * n_examples * self.c_errors) * np.linalg.norm(self.weights) ** 2 + \
                               np.mean(np.where(((np.dot(data, self.weights) + self.biais) * targets) < 1, 1, 0) *
                                       (1 - (np.dot(data, self.weights) + self.biais) * targets))
                class_error_rate_array.append(class_error_rate)
                average_loss_array.append(average_loss)
            #   print "classification error rate: ", class_error_rate
            #   print "average loss: ", average_loss

        if display_errors:
            bins = np.arange(epochs)
            classification_plot, = pylab.plot(bins, class_error_rate_array)
            average_loss_plot, = pylab.plot(bins, average_loss_array)
            pylab.xlabel('epoch')
            pylab.title('Learning curves for the classification error rate and the average loss')
            pylab.legend([classification_plot, average_loss_plot], ['classification error rate', 'average loss'])
            pylab.show()

    def compute_predictions(self, test_data):
        sorties = []

        for i in range(len(test_data)):
            data = []
            for j in range(len(test_data[i])):
                data.append(test_data[i][j])
            sorties.append(np.dot(data, self.weights) + self.biais)
        return sorties