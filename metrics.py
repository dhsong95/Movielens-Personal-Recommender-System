def n_precision(recommendations, hidden, n=100):
    precison = 0.0
    for user, recommendation in enumerate(recommendations):
        counter = 0
        for item in recommendation:
            counter += int(hidden[user, item] == 1)
        precison += (counter / 100)
    N_user = recommendations.shape[0]
    precison /= N_user

    return precison


def n_recall(recommendations, hidden, n=100):
    recall = 0.0
    for user, recommendation in enumerate(recommendations):
        counter = 0
        for item in hidden[user, :].nonzero()[1]:
            counter += int((item == 1) and (item in recommendation))
        recall += (counter / 100)
    N_user = recommendations.shape[0]
    recall /= N_user

    return recall
