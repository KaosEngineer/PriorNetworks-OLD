import os
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma, psi, gammaln
from scipy.stats import dirichlet, pearsonr
from sklearn.metrics import auc
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy.stats import entropy
import seaborn as sns
from scipy.stats import norm
from scipy.stats import gennorm
from scipy.stats import t

sns.set()

# from sklearn.metrics.ranking import
from scipy.stats import spearmanr as rho
from scipy.stats import kendalltau as tau


# from scipy.stats import weightedtau as wtau

### General



def regression_calibration_curve(targets, preds, intervals, save_path):
    diff = np.squeeze(abs(targets - preds))[:, np.newaxis]
    a = np.asarray(diff < intervals, dtype=np.float32)
    emp_frac = np.mean(a, axis=0)
    fraction = np.arange(0.0, 1.0, 0.01)
    plt.close()
    fig, ax = plt.subplots()
    plt.plot(fraction, emp_frac)
    plt.plot(fraction, fraction)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Empirical Fraction')
    plt.xlabel('Fraction')
    plt.xlim(0.0, 1.0)
    plt.savefig(os.path.join(save_path, 'Calibration Curve'), bbox_inches='tight')
    plt.close()


def norm_intervals(means, vars):
    means = np.squeeze(means)
    vars = np.squeeze(vars)
    intervals = np.arange(0.0, 1.0, 0.01)
    return np.asarray(
        [[norm.interval(alpha=interval, loc=0.0, scale=np.sqrt(var))[1] for interval in intervals] for mean, var in
         zip(means, vars)])


def t_intervals(means, vars, nus):
    means = np.squeeze(means)
    vars = np.squeeze(vars)
    nus = np.squeeze(nus)
    intervals = np.arange(0.0, 1.0, 0.01)
    return np.asarray(
        [[t.interval(alpha=interval, df=nu, loc=0.0, scale=np.sqrt(var))[1] for interval in intervals] for mean, var, nu
         in zip(means, vars, nus)])


def gennorm_intervals(means, vars, betas):
    means = np.squeeze(means)
    vars = np.squeeze(vars)
    betas = np.squeeze(betas)
    intervals = np.arange(0.0, 1.0, 0.01)
    return np.asarray(
        [[gennorm.interval(alpha=interval, beta=beta, loc=0.0, scale=np.sqrt(var))[1] for interval in intervals] for
         mean, var, beta in zip(means, vars, betas)])


def norm_calibration_curve(targets, means, log_vars, save_path):
    vars = np.exp(log_vars)
    intervals = norm_intervals(means, vars)
    regression_calibration_curve(targets, means, intervals, save_path)


def td_calibration_curve(targets, means, log_vars, log_nus, save_path):
    vars = np.exp(log_vars)
    nus = np.exp(log_nus)
    intervals = t_intervals(means, vars, nus)
    regression_calibration_curve(targets, means, intervals, save_path)


def ti_calibration_curve(targets, means, log_vars, log_kappas, log_nus, save_path):
    vars = np.exp(log_vars)
    kappas = np.exp(log_kappas)
    nus = np.exp(log_nus)
    vars = (kappas + 1) / (kappas * nus) * vars

    intervals = t_intervals(means, vars, nus)
    regression_calibration_curve(targets, means, intervals, save_path)


def gennorm_calibration_curve(targets, means, log_vars, log_betas, save_path):
    vars = np.exp(log_vars)
    betas = np.exp(log_betas)
    intervals = gennorm_intervals(means, vars, betas)
    regression_calibration_curve(targets, means, intervals, save_path)


def uncertainity_curve(metric, y_pred, y_pred_std, y, save_path, calculate_random_backoff=False):
    y_std_rank = np.flip(y_pred_std.argsort(axis=0), axis=0)
    y_mse_rank = np.flip(np.mean(np.square(y - y_pred), axis=1).argsort(axis=0), axis=0)
    y_pred_ranked_model = y_pred[y_std_rank].reshape(-1, 1)
    y_ranked_model = y[y_std_rank].reshape(-1, 1)
    y_pred_ranked_opt = y_pred[y_mse_rank].reshape(-1, 1)
    y_ranked_opt = y[y_mse_rank].reshape(-1, 1)

    pcc_init = pearsonr(y_ranked_model, y_pred_ranked_model)[0]
    mse_init = -np.mean(np.square(y_ranked_model - y_pred_ranked_model), axis=0)

    pcc_model = np.array([pcc_init])
    mse_model = np.array([mse_init])

    for i in range(len(y)):
        y_back_off_model = np.concatenate((y_ranked_model[0:i + 1], y_pred_ranked_model[i + 1:len(y_pred)]))
        pcc_model = np.append(pcc_model, pearsonr(y_ranked_model, y_back_off_model)[0])
        mse_model = np.append(mse_model, -np.mean(np.square(y_ranked_model - y_back_off_model)))

    pcc_opt = np.array([pcc_init])
    mse_opt = np.array([mse_init])

    for i in range(len(y)):
        y_back_off_opt = np.concatenate((y_ranked_opt[0:i + 1], y_pred_ranked_opt[i + 1:len(y_pred)]))
        pcc_opt = np.append(pcc_opt, pearsonr(y_ranked_opt, y_back_off_opt)[0])
        mse_opt = np.append(mse_opt, -np.mean(np.square(y_ranked_opt - y_back_off_opt)))

    pcc_random = np.array([pcc_init])
    mse_random = np.array([mse_init])

    if calculate_random_backoff == True:

        for i in range(len(y)):
            pcc_tmp = np.array([])
            mse_tmp = np.array([])
            for j in range(200):
                random_index = np.random.permutation(len(y))
                y_ranked_random = y[random_index, :]
                y_pred_ranked_random = y_pred[random_index, :]
                y_back_off_random = np.concatenate((y_ranked_random[0:i + 1], y_pred_ranked_random[i + 1:len(y_pred)]))
                pcc_tmp = np.append(pcc_tmp, pearsonr(y_ranked_random, y_back_off_random)[0])
                mse_tmp = np.append(mse_tmp, -np.mean(np.square(y_ranked_random - y_back_off_random)))
            pcc_random = np.append(pcc_random, np.mean(pcc_tmp))
            mse_random = np.append(mse_random, np.mean(mse_tmp))
    else:
        pcc_random = np.linspace(pcc_model[0], 1, y.shape[0] + 1)
        mse_random = np.linspace(mse_model[0], 0, y.shape[0] + 1)

    if metric == "mse":
        auc_random = np.trapz(np.linspace(0, 1, len(y) + 1), mse_random)
        auc_model = np.trapz(np.linspace(0, 1, len(y) + 1), mse_model)
        auc_expert = np.trapz(np.linspace(0, 1, len(y) + 1), mse_opt)

    elif metric == "pcc":
        auc_random = np.trapz(np.linspace(0, 1, len(y) + 1), pcc_random)
        auc_model = np.trapz(np.linspace(0, 1, len(y) + 1), pcc_model)
        auc_expert = np.trapz(np.linspace(0, 1, len(y) + 1), pcc_opt)

    AUC_RR = (auc_model - auc_random) / float(auc_expert - auc_random)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('MARCO ' + metric + ' ROC using VAR: ' + str(np.round(AUC_RR * 100.0, 1)) + '\n')


def uncertainty_error_correlation_plot(targets, preds, measure, measure_name, pos_label=1, save_path=None):
    error = (np.squeeze(targets) - np.squeeze(preds)) ** 2
    if pos_label != 1:
        measure_loc = -1.0 * measure
    else:
        measure_loc = measure
    measure_loc = np.squeeze(measure_loc)
    sns.regplot(error, measure_loc, fit_reg=False)
    plt.xlabel('L2 Error')
    plt.ylabel(measure_name)
    plt.savefig(os.path.join(save_path, 'Error vs. ' + measure_name + '.png'), bbox_inches='tight')
    plt.close()


def reject_MSE(targets, preds, measure, measure_name, save_path, pos_label=1, show=True):
    if pos_label != 1:
        measure_loc = -1.0 * measure
    else:
        measure_loc = measure
    preds = np.squeeze(preds)
    # Compute total MSE
    error = (preds - targets) ** 2
    MSE_0 = np.mean(error)
    # print 'BASE MSE', MSE_0

    # Create array
    array = np.concatenate(
        (preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)

    # Results arrays
    results_max = [[0.0, 0.0]]
    results_var = [[0.0, 0.0]]
    results_min = [[0.0, 0.0]]

    optimal_ranking = array[:, 2].argsort()
    sorted_array = array[optimal_ranking]  # Sort by error

    for i in xrange(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        # Best rejection
        results_max.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])
        # Random Rejection
        results_min.append([float(i) / float(array.shape[0]), float(i) / float(array.shape[0])])

    uncertainty_ranking = array[:, 3].argsort()
    sorted_array = array[uncertainty_ranking]  # Sort by uncertainty

    for i in xrange(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        results_var.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])

    max_auc = auc([x[0] for x in results_max], [x[1] for x in results_max], reorder=True)
    var_auc = auc([x[0] for x in results_var], [x[1] for x in results_var], reorder=True)
    min_auc = auc([x[0] for x in results_min], [x[1] for x in results_min], reorder=True)

    plt.scatter([x[0] for x in results_max], [x for x in np.asarray(sorted(measure_loc, reverse=True))])
    plt.xlim(0.0, 1.0)
    plt.savefig(os.path.join(save_path, measure_name), bbox_inches='tight')
    plt.close()
    plt.plot([x[0] for x in results_max], [x[1] for x in results_max], 'b^',
             [x[0] for x in results_var], [x[1] for x in results_var], 'ro',
             [x[0] for x in results_min], [x[1] for x in results_min], 'go')
    plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'], loc=4, prop={'size': 18.5})
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Rejection Fraction')
    plt.ylabel('Pearson Correlation')
    if show:
        plt.savefig(os.path.join(save_path, "MSE Rejection Curve using " + measure_name), bbox_inches='tight')
    plt.close()

    AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('MSE ROC using ' + measure_name + ": " + str(np.round(AUC_RR * 100.0, 1)) + '\n')


def reject_PCC(targets, preds, measure, measure_name, save_path, pos_label=1, show=True):
    if pos_label != 1:
        measure_loc = -1.0 * measure
    else:
        measure_loc = measure
    preds = np.squeeze(preds)
    # Compute total MSE and PCC
    error = (preds - targets) ** 2
    P_0 = pearsonr(preds, targets)[0]
    # print 'BASE MSE', P_0
    # Create array
    array = np.concatenate(
        (preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)

    # Results arrays
    results_max = [[0.0, P_0]]
    results_var = [[0.0, P_0]]
    results_min = [[0.0, P_0]]

    optimal_ranking = array[:, 2].argsort()
    sorted_array = array[optimal_ranking]  # Sort by error

    for i in xrange(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        p = pearsonr(x, sorted_array[:, 1])[0]
        # Best rejection
        results_max.append([float(i) / float(array.shape[0]), p])
        # Random Rejection
        results_min.append([float(i) / float(array.shape[0]), P_0 + (1.0 - P_0) * float(i) / float(array.shape[0])])

    uncertainty_ranking = array[:, 3].argsort()
    sorted_array = array[uncertainty_ranking]  # Sort by uncertainty

    for i in xrange(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        p = pearsonr(x, sorted_array[:, 1])[0]
        results_var.append([float(i) / float(array.shape[0]), p])

    # print 'Rho', rho(optimal_ranking, uncertainty_ranking)
    # print 'tau', tau(optimal_ranking, uncertainty_ranking)
    max_auc = auc([x[0] for x in results_max], [x[1] - P_0 for x in results_max], reorder=True)
    var_auc = auc([x[0] for x in results_var], [x[1] - P_0 for x in results_var], reorder=True)
    min_auc = auc([x[0] for x in results_min], [x[1] - P_0 for x in results_min], reorder=True)

    plt.scatter([x[0] for x in results_max], [x for x in np.asarray(sorted(measure_loc, reverse=True))])
    plt.xlim(0.0, 1.0)
    if show:
        plt.savefig(os.path.join(save_path, measure_name), bbox_inches='tight')
    plt.close()
    plt.plot([x[0] for x in results_max], [x[1] for x in results_max], 'b^',
             [x[0] for x in results_var], [x[1] for x in results_var], 'ro',
             [x[0] for x in results_min], [x[1] for x in results_min], 'go')
    plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'], loc=4, prop={'size': 18.5})
    plt.xlim(0.0, 1.0)
    plt.ylim(P_0, 1.0)
    plt.xlabel('Rejection Fraction')
    plt.ylabel('Pearson Correlation')
    if show:
        plt.savefig(os.path.join(save_path, "PCC Rejection Curve using " + measure_name), bbox_inches='tight')
    plt.close()

    AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('PCC ROC using ' + measure_name + ": " + str(np.round(AUC_RR * 100.0, 1)) + '\n')


def plot_pr_curve_id(class_labels, class_probs, measure, measure_name, save_path, pos_label=1, show=True):
    measure = np.asarray(measure, dtype=np.float128)[:, np.newaxis]
    min_measure = np.min(measure)
    if min_measure < 0.0: measure += abs(min_measure)
    measure = np.log(measure + 1e-8)

    class_probs = np.asarray(class_probs, dtype=np.float64)
    class_preds = np.argmax(class_probs, axis=1)[:, np.newaxis]
    class_labels = class_labels[:, np.newaxis]

    rightwrong = np.asarray(class_labels != class_preds, dtype=np.int32)

    if pos_label != 1: measure *= -1.0

    precision, recall, thresholds = precision_recall_curve(rightwrong, measure)
    aupr = auc(recall, precision)
    np.round(aupr, 4)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('AUPR using ' + measure_name + ": " + str(np.round(aupr * 100.0, 1)) + '\n')

    np.savetxt(os.path.join(save_path, measure_name + '_recall_id.txt'), recall)
    np.savetxt(os.path.join(save_path, measure_name + '_precision_id.txt'), precision)

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    if show:
        save_path = os.path.join(save_path, 'PR_' + measure_name + '.png')
        plt.savefig(save_path)
    plt.close()


def plot_roc_curve_id(class_labels, class_probs, measure, measure_name, save_path, pos_label=1, show=True):
    measure = np.asarray(measure, dtype=np.float128)[:, np.newaxis]
    min_measure = np.min(measure)
    if min_measure < 0.0: measure += abs(min_measure)
    measure = np.log(measure + 1e-8)

    class_probs = np.asarray(class_probs, dtype=np.float64)
    class_preds = np.argmax(class_probs, axis=1)[:, np.newaxis]
    class_labels = class_labels[:, np.newaxis]

    rightwrong = np.asarray(class_labels != class_preds, dtype=np.int32)

    if pos_label != 1: measure *= -1.0

    fpr, tpr, thresholds = roc_curve(rightwrong, measure)
    roc_auc = roc_auc_score(rightwrong, measure)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('ROC using ' + measure_name + ": " + str(np.round(roc_auc * 100.0, 1)) + '\n')

    np.savetxt(os.path.join(save_path, measure_name + '_tpr_id.txt'), tpr)
    np.savetxt(os.path.join(save_path, measure_name + '_fpr_id.txt'), fpr)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    if show:
        save_path = os.path.join(save_path, 'PR_' + measure_name + '.png')
        plt.savefig(save_path)
    plt.close()


def plot_accuracy(class_labels, class_probs, measure, measure_name, save_path, pos_label=1, show=True):
    measure = np.asarray(measure, dtype=np.float64)[:, np.newaxis]
    if pos_label != 1:
        measure *= -1.0

    class_probs = np.asarray(class_probs, dtype=np.float64)
    base_accuracy = jaccard_similarity_score(class_labels, np.argmax(class_probs, axis=1))
    class_preds = np.argmax(class_probs, axis=1)[:, np.newaxis]
    class_labels = class_labels[:, np.newaxis]
    rightwrong = np.asarray(class_labels == class_preds, dtype=np.int32)

    accuracies = np.zeros_like(measure, dtype=np.float32)
    accuracies_best = np.zeros_like(measure, dtype=np.float32)
    rejection_fraction = np.arange(0.0, 1.0, 1.0 / float(measure.shape[0]))
    if rejection_fraction.shape[0] > accuracies.shape[0]:
        rejection_fraction = rejection_fraction[:-1]
    elif rejection_fraction.shape[0] < accuracies.shape[0]:
        rejection_fraction = np.concatenate(rejection_fraction, np.asarray([1.0]), axis=0)
    data = np.concatenate((measure, class_labels, class_preds, rightwrong), axis=1)
    data = data[data[:, 0].argsort()]
    data_oracle = data[data[:, 3].argsort()]
    data = data[::-1]

    for i in xrange(measure.shape[0]):
        new_data = np.concatenate((data[:i, 2], data[i:, 1]), axis=0)
        accuracies[i] = jaccard_similarity_score(new_data, data[:, 2])
        new_data = np.concatenate((data_oracle[:i, 2], data_oracle[i:, 1]), axis=0)
        accuracies_best[i] = jaccard_similarity_score(new_data, data_oracle[:, 2])

    rejection_fraction[0] = 0.0
    rejection_fraction[-1] = 1.0
    accuracies[-1] = 1.0
    accuracies_best[-1] = 1.0
    accuracies_best[0] = base_accuracy
    accuracies[0] = base_accuracy

    max_area = auc(rejection_fraction, accuracies_best) - base_accuracy - 0.5 * (1 - base_accuracy)
    area_under_curve = auc(rejection_fraction, accuracies) - base_accuracy - 0.5 * (1 - base_accuracy)
    auc_ratio = area_under_curve / max_area
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('AUC Ratio using ' + measure_name + ': ' + str(auc_ratio) + '\n')

    plt.plot(rejection_fraction, accuracies)
    plt.plot(rejection_fraction, accuracies_best)
    plt.legend(['Uncertainty Rejection', 'Optimal Rejection'], loc=4)
    plt.ylabel('Accuracy')
    plt.ylim(min(accuracies), 1.0)
    plt.xlim(0.0, 1.0)
    plt.xlabel('Rejection Fraction')
    if show:
        save_path = os.path.join(save_path, 'Rejection_Curve_' + measure_name + '.png')
        plt.savefig(save_path)
    plt.close()


def plot_accuracies(class_labels, class_probs, uncertainties, save_path, show=False):
    functions = [plot_roc_curve_id, plot_pr_curve_id]
    for function in functions:
        for key in uncertainties.keys():
            if key == 'max_prob':
                pos_label = 0
            else:
                pos_label = 1
            function(class_labels, class_probs, uncertainties[key][0], uncertainties[key][1],
                     save_path=save_path, show=show, pos_label=pos_label)


def plot_rejection(targets, preds, uncertainties, save_path, show=False):
    for key in uncertainties.keys():
        if key == 'max_prob':
            pos_label = 0
        else:
            pos_label = 1
        reject_PCC(targets, preds, uncertainties[key][0], uncertainties[key][1],
                   save_path=save_path, show=show, pos_label=pos_label)
        reject_MSE(targets, preds, uncertainties[key][0], uncertainties[key][1],
                   save_path=save_path, show=show, pos_label=pos_label)
        uncertainty_error_correlation_plot(targets, preds, uncertainties[key][0], uncertainties[key][1],
                                           pos_label=pos_label, save_path=save_path)


def plot_roc_curve(domain_labels, in_measure, out_measure, measure_name, save_path, pos_label=1, show=True):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    print measure_name
    print np.isnan(np.sum(scores)), np.max(scores), np.min(scores)
    if pos_label != 1:
        scores *= -1.0

    fpr, tpr, thresholds = roc_curve(domain_labels, scores)
    roc_auc = roc_auc_score(domain_labels, scores)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('ROC AUC using ' + measure_name + ": " + str(np.round(roc_auc * 100.0, 1)) + '\n')
    np.savetxt(os.path.join(save_path, measure_name + '_trp.txt'), tpr)
    np.savetxt(os.path.join(save_path, measure_name + '_frp.txt'), fpr)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    if show:
        save_path = os.path.join(save_path, 'ROC_' + measure_name + '.png')
        plt.savefig(save_path)
    plt.close()


def plot_pr_curve(domain_labels, in_measure, out_measure, measure_name, save_path, pos_label=1, show=True):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    if pos_label != 1:
        scores *= -1.0

    precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
    aupr = auc(recall, precision)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write(
            'AUPR using ' + measure_name + ": " + str(np.round(aupr * 100.0, 1)) + '\n')
    np.savetxt(os.path.join(save_path, measure_name + '_recall_ood.txt'), recall)
    np.savetxt(os.path.join(save_path, measure_name + '_precision_ood.txt'), precision)

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    if show:
        save_path = os.path.join(save_path, 'PR_' + measure_name + '.png')
        plt.savefig(save_path)
    plt.close()


def plot_histogram(uncertainty_measure, measure_name, ood_uncertainty_measure, save_path=None, log=False, show=True,
                   bins=50):
    uncertainty_measure = np.asarray(uncertainty_measure, dtype=np.float128)
    ood_uncertainty_measure = np.asarray(ood_uncertainty_measure, dtype=np.float128)
    # print measure_name
    if log == True:
        scores = np.concatenate((uncertainty_measure, ood_uncertainty_measure), axis=0)
        min_score = np.min(scores)
        if min_score < 0.0:
            uncertainty_measure += abs(min_score)
            ood_uncertainty_measure += abs(min_score)
        uncertainty_measure = np.log(uncertainty_measure + 1.0)
        ood_uncertainty_measure = np.log(ood_uncertainty_measure + 1.0)
    scores = np.concatenate((uncertainty_measure, ood_uncertainty_measure), axis=0)
    min_score = np.min(scores)
    max_score = np.max(scores)
    # print min_score, max_score
    plt.hist(uncertainty_measure, bins=bins / 2, range=(min_score, max_score),alpha=0.5)
    plt.xlabel(measure_name)

    plt.hist(ood_uncertainty_measure, bins=bins / 2, range=(min_score, max_score),alpha=0.5)
    plt.legend(['In-Domain', 'Out-of-Domain'])

    plt.xlim(min_score, max_score)
    if show:
        save_path = os.path.join(save_path, 'Histogram_' + measure_name + '.png')
        plt.savefig(save_path)
    plt.close()


def plot_roc_curves(domain_labels, in_uncertainties, out_uncertainties, save_path, log=False, show=True, classes_flipped=None, adversarial=False):
    functions = [plot_roc_curve, plot_pr_curve]
    if adversarial:
        functions = [plot_mod_roc_curve]
    for function in functions:
        for key in in_uncertainties.keys():
            if key == 'max_prob':
                pos_label = 0
            else:
                pos_label = 1
            if adversarial:
                function(domain_labels, in_uncertainties[key][0], out_uncertainties[key][0], classes_flipped,
                         in_uncertainties[key][1],
                         save_path=save_path, pos_label=pos_label, show=show)
            else:
                function(domain_labels, in_uncertainties[key][0], out_uncertainties[key][0],
                         in_uncertainties[key][1],
                         save_path=save_path, pos_label=pos_label, show=show)



def plot_uncertainties(uncertainties, ood_uncertainties, save_path=None, log=False, show=True):
    for key in uncertainties.keys():
        if key == 'max_prob':
            pos_label = 0
        else:
            pos_label = 1
        plot_histogram(uncertainties[key][0], uncertainties[key][1], ood_uncertainties[key][0], log=log,
                       save_path=save_path, show=show)


def mod_roc_curve(y_true, y_score, class_flipped, pos_label=1):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
        class_flipped : array, shape = [n_samples]
        whether the predicted class was flipped or not
    pos_label : int or str, default=None
        The label of the positive class
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """

    # y_true = column_or_1d(y_true)
    # y_score = column_or_1d(y_score)

    # make y_true a boolean vector
    #y_true = (y_true == pos_label)
    print y_true

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    class_flipped = class_flipped[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    tpr = tps / np.float(tps[-1])


    y_true = 1- y_true
    total_neg = np.float(np.sum(y_true))
    y_true = y_true*class_flipped
    print total_neg, np.sum(y_true)
    fps = np.cumsum(y_true)[threshold_idxs]
    fpr = fps / total_neg


    fpr = np.r_[fpr, 1.0]
    tpr = np.r_[tpr, 1.0]

    auc_score = auc(fpr, tpr, reorder=True)

    return auc_score, fpr, tpr, y_score[threshold_idxs]


def plot_mod_roc_curve(domain_labels, in_measure, out_measure, class_flipped, measure_name, save_path, pos_label=1, show=True):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    not_flipped = np.asarray(np.zeros_like(in_measure), dtype=np.int32)
    class_flipped = np.r_[not_flipped, class_flipped]
    print measure_name
    print np.isnan(np.sum(scores)), np.max(scores), np.min(scores)
    if pos_label != 1:
        scores *= -1.0

    roc_auc, fpr, tpr, thresholds = mod_roc_curve(domain_labels, scores, class_flipped=class_flipped)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('ROC AUC using ' + measure_name + ": " + str(np.round(roc_auc * 100.0, 1)) + '\n')
    np.savetxt(os.path.join(save_path, measure_name + '_trp.txt'), tpr)
    np.savetxt(os.path.join(save_path, measure_name + '_frp.txt'), fpr)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    if show:
        save_path = os.path.join(save_path, 'ROC_' + measure_name + '.png')
        plt.savefig(save_path)
    plt.close()
### Dirichlet Network


def calc_dirichlet_expected_entropy(alphas, epsilon=1e-8):
    # Calculate Expected Entropy of categorical distribution under dirichlet Prior.
    # Higher means more uncertain
    alphas = np.asarray(alphas, dtype=np.float64) + epsilon
    alpha_0 = np.sum(alphas, axis=1, keepdims=True)
    expected_entropy = -np.sum(
        np.exp(np.log(alphas) - np.log(alpha_0)) * (digamma(alphas + 1.0) - digamma(alpha_0 + 1.0)), axis=1)
    return expected_entropy


def calc_dirichlet_differential_entropy(alphas, epsilon=1e-8):
    # Calculate Expected Entropy of categorical distribution under dirichlet Prior.
    # Higher means more uncertain
    alphas = np.asarray(alphas, dtype=np.float64) + epsilon
    diff_entropy = np.asarray([dirichlet(alpha).entropy() for alpha in alphas])
    return diff_entropy


def calc_dirichlet_entropy_expected(probs, epsilon=1e-8):
    # Calculate Entropy of  Expected categorical distribution under dirichlet Prior.
    # Higher means more uncertain
    probs = np.asarray(probs, dtype=np.float64) + epsilon
    entropy_expected = np.asarray([entropy(prob) for prob in probs], dtype=np.float64)
    # log_probs = np.log(probs)
    # entropy_expected=-np.squeeze(np.sum(probs*log_probs, axis=1))
    return entropy_expected


def calc_dirichlet_max_predicted_prob(probs, epsilon=1e-8):
    # Return highest probability. Higher means less uncertain.

    max_pred_prob = np.max(probs, axis=1)
    return max_pred_prob


def calc_dirichlet_variation_ratio(probs, epsilon=1e-8):
    return 1.0 - calc_dirichlet_max_predicted_prob(probs, epsilon)


def calc_dirichlet_mutual_information(probs, alphas, epsilon=1e-8):
    # Calculate Entropy of  Expected categorical distribution under dirichlet Prior.
    # Higher means more uncertain

    exp_ent = calc_dirichlet_expected_entropy(alphas, epsilon=epsilon)
    ent_exp = calc_dirichlet_entropy_expected(probs, epsilon=epsilon)
    # alphas = np.asarray(alphas, dtype=np.float64) + epsilon
    # alpha_0 = np.sum(alphas,axis=1,keepdims=True)
    # mi = -np.sum(alphas/alpha_0*(np.log(alphas)-np.log(alpha_0)-psi(alphas+1)+psi(alpha_0+1)),axis=1)
    # mi = ent_exp-exp_ent
    return ent_exp  # -exp_ent


def calc_dirichlet_expected_pairwise_KL(alphas, epsilon=1e-8):
    # Return Expected Divergence Between Pairs of samples from dirchlet
    # Higher Means More Uncertain

    K = alphas.shape[1]
    epkl = (K - 1.0) / np.sum(alphas, axis=1)

    return epkl


def calculate_dirichlet_uncertainty(probs, alphas, epsilon=1e-8):
    entropy_expected = calc_dirichlet_entropy_expected(probs)
    max_prob = calc_dirichlet_max_predicted_prob(probs)
    #epkl = calc_dirichlet_expected_pairwise_KL(alphas)
    mi = calc_dirichlet_mutual_information(probs, alphas, epsilon)
    diff_ent = calc_dirichlet_differential_entropy(alphas)
    #var_ratio = calc_dirichlet_variation_ratio(probs)

    return {'entropy_expected': [entropy_expected, 'Entropy_of_Expected_Distribution'],
            'max_prob': [max_prob, 'Max_Predicted_Probability'],
            # 'epkl' : [epkl, 'Expected_Pairwise_KL_Divergence'],
            'mutual_information': [mi, 'Mututal_Information'],
            'diffential_entropy': [diff_ent, 'Differential_Entropy']}
    # 'variation_ratio' : [var_ratio, 'Variation_Ratio']}


### Normal inverse Wishart uncertainty measures

def calc_niw_differential_entropy(log_var, log_kappa, log_nu, epsilon=1e-8):
    # Higher means more uncertain
    nu = np.exp(log_nu)
    nu = np.squeeze(nu)
    log_kappa = np.squeeze(log_kappa)
    log_var = np.squeeze(log_var)

    diff_ent = 0.5 * (log_var - np.log(2.0) - psi(nu / 2.0) + 1 - log_kappa + np.log(2.0 * np.pi)) \
               + gammaln(nu / 2.0) + nu / 2.0 + (log_var - np.log(2.0)) - (nu - 2.0) / 2.0 * psi(nu / 2.0)

    return diff_ent


def calc_niw_expected_differential_entropy(log_var, log_kappa, log_nu, epsilon=1e-8):
    D = log_var.shape[1]
    # var = np.exp(log_var)
    kappa = np.exp(log_kappa)
    nu = np.exp(log_nu)
    # var = (kappa+1.0)/(kappa*nu)*var

    entropy = 0.5 * np.sum(log_var, axis=1, keepdims=True) + (nu + 1.0) / 2.0 * (
    psi((nu + 1.0) / 2.0) - psi((nu - D + 1.0) / 2.0)) + \
              D / 2.0 * np.log((kappa + 1.0) / (kappa * (nu - D + 1))) + D / 2.0 * np.log(np.pi * (nu - D + 1.0)) + \
              gammaln((nu - D + 1.0) / 2.0) - gammaln((nu + 1.0) / 2.0)

    return np.squeeze(entropy)
    # return gammaln(nu/2.0)+0.5*np.log(nu*np.pi)-gammaln((nu+1.0)/2.0)+0.5*log_var


def calc_students_variance(log_var, log_nu, epsilon=1e-8):
    nu = np.exp(log_nu)
    return np.exp(log_var) * nu / (nu + 2.0)


def calc_students_mode_probability(log_var, log_nu, epsilon=1e-8):
    var = np.exp(log_var)
    nu = np.exp(log_nu)
    D = log_var.shape[1]

    return gamma((nu + D) / 2.0) / (
    gamma(nu / 2.0) * np.sqrt(np.prod(var, axis=1, keepdims=True) * np.power(nu * np.pi, D) + epsilon))


def calc_students_differential_entropy(log_var, log_nu, epsilon=1e-8):
    var = np.exp(log_var)
    nu = np.exp(log_nu)
    D = log_var.shape[1]

    entropy = (nu + D) / 2.0 * (psi((nu + D) / 2.0) - psi(nu / 2.0)) + 0.5 * np.sum(log_var, axis=1,
                                                                                    keepdims=True) + gammaln(nu / 2.0) \
              - gammaln((nu + D) / 2.0) + D / 2.0 * np.log(nu * np.pi)

    return np.squeeze(entropy)


def calc_students_variance_of_expected(log_var, log_kappa, log_nu, epsilon=1e-8):
    kappa = np.exp(log_kappa)
    nu = np.exp(log_nu)

    return np.exp(log_var) * (kappa + 1.0) / (kappa * (nu - 2.0))


def calc_niw_expected_mode_probability(log_var, log_kappa, log_nu, epsilon=1e-8):
    var = np.exp(log_var)
    kappa = np.exp(log_kappa)
    nu = np.exp(log_nu)
    var = (kappa + 1.0) / (kappa * nu) * var

    return gamma((nu + 1.0) / 2.0) / (gamma(nu / 2.0) * np.sqrt(1e-8 + nu * np.pi * var))


def calc_niw_mutual_information(log_var, log_kappa, log_nu):
    nu = np.exp(log_nu)
    mi = calc_niw_expected_differential_entropy(log_var, log_kappa, log_nu)[:, np.newaxis]
    mi -= 0.5 * (1.0 + np.log(np.pi) + log_var - psi(nu / 2.0))

    return mi


def calculate_niw_uncertainty(log_vars, log_kappa, log_nu):
    dentropy_niw = calc_niw_differential_entropy(log_vars, log_kappa, log_nu)
    dentropy_t = calc_niw_expected_differential_entropy(log_vars, log_kappa, log_nu)
    var_t = calc_students_variance_of_expected(log_vars, log_kappa, log_nu)
    mode_prob = calc_niw_expected_mode_probability(log_vars, log_kappa, log_nu)
    mi = calc_niw_mutual_information(log_vars, log_kappa, log_nu)

    return {'dentropy_t': [np.squeeze(dentropy_t), 'dentropy_t'],
            'dentropy_niw': [np.squeeze(dentropy_niw), 'dentropy_niw'],
            'var_t': [np.squeeze(var_t), 'var_t'],
            'max_prob': [np.squeeze(mode_prob), 'max_prob'],
            'mutual_information': [np.squeeze(mi), 'mutual_information']}


### Multivariate Normal uncertainty measures

def calc_mvn_differential_entropy(log_var, epsilon=1e-8):
    # Higher means more uncertain
    return 0.5 * (log_var + 1 + np.log(2.0 * np.pi))


def calc_mvn_mode_probability(log_var, epsilon=1e-8):
    var = np.minimum(np.exp(log_var), 1e2)
    return 1.0 / np.sqrt(2.0 * np.pi * var)


def calculate_mvn_uncertainty(log_vars, epsilon=1e-8):
    dentropy_t = calc_mvn_differential_entropy(log_vars)

    return {'dentropy': [np.squeeze(dentropy_t), 'dentropy'],
            'var': [np.squeeze(np.minimum(np.exp(log_vars), 1e6)), 'var']}


### Multivariate T uncertainty measures

def calculate_td_uncertainty(log_vars, log_nu, epsilon=1e-8):
    dentropy_t = calc_students_differential_entropy(log_vars, log_nu)
    mode_prob = calc_students_mode_probability(log_vars, log_nu)
    variance = calc_students_variance(log_vars, log_nu)

    return {'dentropy': [np.squeeze(dentropy_t), 'dentropy_t'],
            'var': [np.squeeze(variance), 'var'],
            'max_prob': [np.squeeze(mode_prob), 'max_prob']}


### Generalized Normal Distribution measures

def calc_gnd_differential_entropy(log_alpha, log_beta, epsilon):
    beta = np.exp(log_beta)
    return 1 / beta - log_beta + np.log(2) + log_alpha + gammaln(1 / beta)


def calc_gnd_variance(log_alpha, log_beta, epsilon):
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    return (alpha ** 2.0) * gamma(3 / beta) / gamma(1 / beta)


def calc_gnd_mode_probability(log_alpha, log_beta, epsilon):
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    return beta / (2.0 * alpha * gamma(1 / beta))


def calculate_gnd_uncertainty(log_alpha, log_beta, epsilon=1e-8):
    dentropy_t = calc_gnd_differential_entropy(log_alpha, log_beta, epsilon=epsilon)
    mode_prob = calc_gnd_mode_probability(log_alpha, log_beta, epsilon=epsilon)
    var = calc_gnd_variance(log_alpha, log_beta, epsilon=epsilon)

    return {'dentropy': [np.squeeze(dentropy_t), 'dentropy'],
            'var': [np.squeeze(var), 'var'],
            'max_prob': [np.squeeze(mode_prob), 'max_prob']}


#### Monte-Carlo Dropout Bayesian Uncertainty for Classification

def calc_MCDP_expected_entropy(data, epsilon=1e-8):
    log_data = np.log(data + epsilon)
    expected_entropy = -np.mean(np.sum(data * log_data, axis=2), axis=1)

    return expected_entropy


def calc_MCDP_entropy_expected(data, epsilon=1e-8):
    mean_data = np.mean(data, axis=1)
    log_mean_data = np.log(mean_data + epsilon)

    entropy_expected = -np.squeeze(np.sum(mean_data * log_mean_data, axis=1))

    return entropy_expected


def calc_MCDP_mutual_information(data, epsilon=1e-8):
    entropy_of_expected = calc_MCDP_entropy_expected(data=data, epsilon=epsilon)
    expected_entropy = calc_MCDP_expected_entropy(data=data, epsilon=epsilon)

    mutual_information = entropy_of_expected - expected_entropy
    return mutual_information


def calc_MCDP_max_predicted_prob(data, epsilon=1e-8):
    mean_data = np.mean(data, axis=1)
    max_pred_prob = np.max(mean_data, axis=1)
    return max_pred_prob


def classification_calibration(labels, probs, save_path, bins=10):
    preds = np.argmax(probs, axis=1)
    total = labels.shape[0]
    probs = np.max(probs, axis=1)
    lower = 0.0
    increment = 1.0 / bins
    upper = increment
    accs = np.zeros([bins], dtype=np.float32)
    gaps = np.zeros([bins], dtype=np.float32)
    confs = np.arange(0.0, 1.00, increment)
    ECE = 0.0
    for i in xrange(bins):
        ind1 = probs >= lower
        ind2 = probs < upper
        ind = np.where(np.logical_and(ind1, ind2))[0]
        lprobs = probs[ind]
        lpreds = preds[ind]
        llabels = labels[ind]
        acc = jaccard_similarity_score(llabels, lpreds)
        prob = np.mean(lprobs)
        if np.isnan(acc):
            acc = 0.0
            prob = 0.0
        ECE += np.abs(acc - prob) * float(lprobs.shape[0])
        gaps[i] = np.abs(acc - prob)
        accs[i] = acc
        upper += increment
        lower += increment
    ECE /= np.float(total)
    MCE = np.max(np.abs(gaps))

    fig, ax = plt.subplots()
    plt.plot(confs, accs)
    plt.plot(confs, confs)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.xlim(0.0, 1.0)
    plt.savefig(os.path.join(save_path, 'Reliability Curve'), bbox_inches='tight')
    plt.close()
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('ECE: ' + str(np.round(ECE * 100.0, 2)) + '\n')
        f.write('MCE: ' + str(np.round(MCE * 100.0, 2)) + '\n')


def calc_MCDP_expected_pairwise_KL(probs, epsilon=1e-8):
    expected_entropy = calc_MCDP_expected_entropy(probs)

    log_probs = np.log(probs + epsilon)
    new_data = np.zeros((probs.shape[0], probs.shape[2]), dtype=np.float32)
    for x in xrange(probs.shape[0]):
        for k in xrange(probs.shape[2]):
            new_data[x][k] = np.mean(np.outer(probs[x, :, k], log_probs[x, :, k]))
    epkl = -expected_entropy - np.sum(new_data, axis=1)

    return epkl


def calc_MCDP_variation_ratio(probs):
    labels = np.argmax(probs, axis=2)
    max_labels = [np.argmax([np.sum(label == i) for i in xrange(probs.shape[2])]) for label in labels]
    max_occurs = np.asarray([np.sum(label == max_label) for label, max_label in zip(labels, max_labels)],
                            dtype=np.float32)
    var_ratio = 1.0 - max_occurs / float(probs.shape[1])
    return var_ratio


def calculate_MCDP_uncertainty(probs, log=False):
    entropy_expected = calc_MCDP_entropy_expected(probs)
    max_prob = calc_MCDP_max_predicted_prob(probs)
    # epkl = calc_MCDP_expected_pairwise_KL(probs)
    mi = calc_MCDP_mutual_information(probs)
    var_ratio = calc_MCDP_variation_ratio(probs)

    return {'entropy_expected': [entropy_expected, 'Entropy_of_Expected_Distribution'],
            'max_prob': [max_prob, 'Max_Predicted_Probability'],
            # 'epkl' : [epkl, 'Expected_Pairwise_KL_Divergence'],
            'mutual_information': [mi, 'Mututal_Information'],
            'variation_ratio': [var_ratio, 'Variation_Ratio']}


#### OOD Uncertainty

def calculate_ood_uncertainty(probs):
    max_prob = calc_dirichlet_max_predicted_prob(probs)
    ood_probs = probs[:, -1]

    return {'ood_prob': [ood_probs, 'Probability of OOD Class'],
            'max_prob': [max_prob, 'Max_Predicted_Probability']}
