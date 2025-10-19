from copy import copy

import autograd.numpy as np
from autograd import grad
from src.optimizers import adam
from src.utility_functions import make_batches


class HSBnn:
    def __init__(self, task_id, last_model, layer_sizes, x, y, x_test, y_test, inference_engine, mask, batch_size):
        self.T = task_id
        self.shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.layer_sizes = layer_sizes
        self.N_weights = sum((m+1)*n for m, n in self.shapes)
        self.batches = make_batches(x.shape[0], batch_size)
        self.M = len(self.batches)  # Total number of batches
        self.elbo = list()
        self.val_ll = list()
        self.val_err = list()
        self.train_err = list()
        self.variational_params = None
        self.init_params = None
        self.variational_params_store = {}
        self.optimal_elbo_params = None
        self.X = x
        self.y = y
        self.X_test = x_test
        self.y_test = y_test
        self.mask = mask
        self.task_specific_mask = None
        if task_id == 0:
            self.task_specific_mask_list = {}
        else:
            self.task_specific_mask_list = last_model.task_specific_mask_list
            #self.task_specific_mask[task_id-1] = last_model.mask
            self.task_specific_mask_list[task_id-1] = last_model.task_specific_mask
        self.inference_engine = inference_engine(t=self.T, last_model=last_model, n_weights=self.N_weights,
                                                 shapes=self.shapes)

    def neg_elbo(self, params, epoch, x, y):
        temperature = 1
        log_lik, log_prior, ent_w, ent_tau, ent_lam = self.inference_engine.compute_elbo_contribs(params, x, y, self.mask)
        log_variational = ent_w + ent_tau + ent_lam
        minibatch_rescaling = 1./self.M
        ELBO = temperature * minibatch_rescaling * (log_variational + log_prior) + log_lik
        return -1*ELBO

    def variational_objective(self, params, t):
        idx = self.batches[t % self.M]
        return self.neg_elbo(params, t/self.M, self.X[idx], self.y[idx])

    def compute_optimal_test_ll(self, task_id, X_test, y_test, num_samples=5):
        if task_id == self.T:
            params = copy(self.variational_params)
            return self.inference_engine.compute_test_ll(task_id, params, X_test, y_test,\
                                                         self.task_specific_mask, num_samples=num_samples)
        else:
            params = copy(self.optimal_elbo_params)
            return self.inference_engine.compute_test_ll(task_id, params, X_test, y_test,\
                                                         self.task_specific_mask_list[task_id], num_samples=num_samples)


def fit(model, mask, n_epochs=10, l_rate=0.01, t_id=100):
    def callback(params, t, g, decay=0.999):
        score = -model.variational_objective(params, t)
        model.elbo.append(score)
        if (t % model.M) == 0:
            val_ll, val_err = model.inference_engine.compute_test_ll(model.T, params, model.X_test, model.y_test, model.mask)
            train_err = model.inference_engine.compute_train_err(params, model.X, model.y, model.mask)
            model.val_ll.append(val_ll)
            model.val_err.append(val_err)
            model.train_err.append(train_err)
            if ((t / model.M) % 10) == 0:
                print("Epoch {} lower bound {} train_err {} test_err {} ".format(t/model.M, model.elbo[-1],
                                                                                 train_err,
                                                                                 model.val_err[-1]))

            # randomly permute batch ordering every epoch
            model.batches = np.random.permutation(model.batches)
        if (t % 250) == 0:
            # store optimization progress.
            model.variational_params_store[t] = copy(params)
        if t > 2:
            if model.elbo[-1] > max(model.elbo[:-1]):
                model.optimal_elbo_params = copy(params)
        # update inverse gamma distributions
        model.inference_engine.fixed_point_updates(model.mask, params)

    if model.variational_params is not None:
        init_var_params = copy(model.variational_params)
    else:
        init_var_params = model.inference_engine.initialize_variational_params(t_id)
    model.init_params = copy(init_var_params)
    gradient = grad(model.variational_objective, 0)
    num_iters = n_epochs * model.M  # one iteration = one set of param updates
    model.variational_params = adam(gradient, init_var_params, mask,
                                    step_size=l_rate, num_iters=num_iters, callback=callback)
    return model



def teacher_network_selection(t, model, x_train, y_train):
    min_err = 1.0
    best_tt = -1
    for tt in range(t):
        _, err = model.compute_optimal_test_ll(tt, x_train, y_train)
        print(err)
        if err < min_err:
            min_err = err
            best_tt = tt
    if min_err < 0.25:
        print("Minimum error is {0} @ Task {1}".format(min_err, best_tt))
        return (model.task_specific_mask_list[best_tt], best_tt)
    else:
        print("No appropriate teacher.")
        return None