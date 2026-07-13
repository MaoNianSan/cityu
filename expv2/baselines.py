from formulation import classic_estimate_se, naive_estimate_se


def classic_inference(task, X_labelled, y_labelled):
    return classic_estimate_se(task, X_labelled, y_labelled)


def naive_ml_inference(task, X_unlabelled, pred_unlabelled):
    return naive_estimate_se(task, X_unlabelled, pred_unlabelled)
