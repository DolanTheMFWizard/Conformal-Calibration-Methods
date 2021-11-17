import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from conformal_cal import FixedDiagonalDirichletCalibrator, FullDirichletCalibrator, TemperatureScaling, MatrixScaling, \
    VectorScaling

DIRICHLET_CALIBRATE = 'dirichlet_calibrate'
TEMP_SCALING = 'temperature_scaling'
VECTOR_SCALING = 'vector_scaling'
MATRIX_SCALING = 'matrix_scaling'
FIXED_DIRICHLET = 'fixed_dirichlet'

CONFORMAL_CALIBRATE_LIST = [DIRICHLET_CALIBRATE, TEMP_SCALING, VECTOR_SCALING, MATRIX_SCALING, FIXED_DIRICHLET]


def conformal_calibrate_model(self, calibration_method: str, y_val_pred_proba: np.ndarray, y_val_label: np.ndarray,
                              num_splits: int = 3,
                              reg_list: list = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1]):
    """
    Applies conformal learning method specified by the user

    Parameters:
    -----------
    calibration_method: str
        conformal calibration method to apply
    y_val_pred_proba: np.ndarray
        The predictive probabilities of the validation set for the model you
        are trying to calibrate. Should be in multi-class form. Looking at you
        binary problems.
    y_val_label: np.ndarray
        The labels for the validation set
    NOTE!!!: y_val_label and y_val_label should have classes in nominal form
    num_splits: int: default=3
        Number of splits for StratifiedKFold for dirichlet calibrate full
    reg_list: list: default=[0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1]
        Lambda regularization list for regularization of conformal method

    """
    if calibration_method == DIRICHLET_CALIBRATE:
        calibrator = FullDirichletCalibrator(reg_lambda=reg_list)
    elif calibration_method == TEMP_SCALING:
        calibrator = TemperatureScaling(reg_lambda_list=reg_list)
    elif calibration_method == MATRIX_SCALING:
        calibrator = MatrixScaling(reg_lambda_list=reg_list)
    elif calibration_method == VECTOR_SCALING:
        calibrator = VectorScaling(reg_lambda_list=reg_list)
    elif calibration_method == FIXED_DIRICHLET:
        calibrator = FixedDiagonalDirichletCalibrator()
    else:
        raise Exception(
            f'Given unknown calibration method {calibration_method}. Should be one of: {CONFORMAL_CALIBRATE_LIST}')

    if calibration_method == DIRICHLET_CALIBRATE:
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=0)
        calibrator = GridSearchCV(calibrator, param_grid={'reg_lambda': reg_list,
                                                          'reg_mu': [None]},
                                  cv=skf, scoring='neg_log_loss')
        calibrator.fit(y_val_pred_proba, y_val_label)
    else:
        calibrator.fit(y_val_pred_proba, y_val_label)

    return calibrator
