from tpot import TPOTClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectPercentile, f_classif

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

import pickle

import text

from text import viz, common


def find_best_model(X_train, X_test, y_train, y_test):
    pipeline_optimizer = TPOTClassifier(
        generations=100,
        population_size=50,
        cv=5,
        random_state=42,
        verbosity=2,
        config_dict='TPOT sparse'
    )

    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))

    pipeline_optimizer.export('outputs/tpot_exported_pipeline.py')


def run_all(show_happy_faces=False):
    X_train, X_test, y_train, y_test, tfidf_vectorizer = text.get_prepared_gata()

    clfs = {
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier()
    }

    for clf_name in clfs:
        clf = clfs[clf_name]
        clf.class_weight = 'balanced'
        clf.fit(X_train, y_train)

        predicted = clf.predict(X_test)

        if show_happy_faces:
            print('predicted       truth      accurate?\n' + '-' * 37)
            for text, category, truth in zip(X_test, predicted, y_test):
                print('     {}    |       {}     |     {}'.format(
                    '+' if category else '-',
                    '+' if truth else '-',
                    ':)' if category == truth else ':(')
                )
        print('..-^-'*5 + clf_name + '-^-..'*5)
        viz.show_stats(y_test, predicted)


def fs_svc(X_train, y_train, persist=False):
    # Feature selection before SVC
    print('Feature Selection SVC Model')

    # Score on the training set was:0.6328609072087332
    pipeline = make_pipeline(
        SelectPercentile(score_func=f_classif, percentile=18),
        LinearSVC(C=1.0, dual=True, loss="hinge", penalty="l2", tol=0.001)
    )

    pipeline.fit(X_train, y_train)

    if persist:
        with open('pickles/new_fs_svc.pickle', 'wb') as f:
            pickle.dump(pipeline, f)

    return pipeline


def bnb(X_train, X_test, y_train, y_test, persist_path=None, do_viz=False):
    # Feature selection before SVC
    # print('BeronulliNB')

    pipeline = make_pipeline(
        RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.1, n_estimators=100), step=0.4),
        BernoulliNB(alpha=0.01, fit_prior=False)
    )

    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    if do_viz:
        viz.show_stats(y_test, predicted)

    common.save_pickle(pipeline, persist_path)

    # returns train and test classified
    return pipeline
