import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy.stats import shapiro, pearsonr
from sklearn.metrics import recall_score, make_scorer, accuracy_score, f1_score, mean_squared_error, classification_report, confusion_matrix, multilabel_confusion_matrix, precision_score, roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics.scorer import _BaseScorer
from statistics import pstdev, mean
from typing import Dict, List, ClassVar, Set
from math import sqrt

def optimal_threshold(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def compute_binary_cutoffs(y_true, y_pred):
    if y_true.shape == y_pred.shape and len(y_true.shape) == 1:  # 2 classes
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return [optimal_threshold(fpr, tpr, thresholds)]
    elif y_true.shape == y_pred.shape and len(y_true.shape) == 2:  # multilabel
        fpr_tpr_thresholds = [
            roc_curve(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
        return [optimal_threshold(*x) for x in fpr_tpr_thresholds]


class ClassificationMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 labels: List = None,
                 validation_generator=None,
                 validation_data=None,
                 multi_label=False):
        super().__init__()
        if labels is not None:
            self.labels = {name: index for index, name in enumerate(labels)}
            self.binary = (len(labels) == 2)

        elif validation_generator is not None:
            self.labels = validation_generator.class_indices
            self.binary = False
        self.validation_generator = validation_generator
        self.validation_data = validation_data
        self.multi_label = multi_label
        self._binary_cutoffs = []
        self._data = []
        

    def on_epoch_end(self, batch, logs={}):
        if self.validation_generator is None:
            X_val, y_val = self.validation_data[0], self.validation_data[1]
            y_pred = np.asarray(self.model.predict(X_val))
            

        else:
            y_pred = []
            y_val = []
            for i in range(len(self.validation_generator)):
                xVal, yVal = self.validation_generator[i]
                if type(yVal) is list:
                    y_val.append(np.stack(yVal, axis=-1))
                    y_pred.append(
                        np.stack(np.squeeze(self.model.predict(xVal)),
                                 axis=-1))
                elif type(yVal) is dict:
                    y_val.append(yVal)
                    task_preds = self.model.predict(xVal)
                    y_pred.append({
                        k: task_preds[i]
                        for i, k in enumerate(
                            self.validation_generator.task_names)
                    })

                else:
                    y_val.append(np.asarray(yVal))
                    y_pred.append(np.asarray(self.model.predict(xVal)))
                    if np.isnan(xVal).any():
                        print('NaN in input!')

        if type(y_pred) == list:
            y_pred = np.concatenate(y_pred)
        nans_in_predictions = np.argwhere(np.isnan(y_pred))
        if type(y_val) == list:
            y_val = np.concatenate(y_val)
        
        nans_in_validation = np.argwhere(np.isnan(y_val))
        if nans_in_predictions.any() or nans_in_validation.any():
            print(nans_in_predictions, nans_in_validation,
                    len(nans_in_predictions), len(nans_in_validation))

        logs = self.compute_metrics(y_val,
                                    y_pred,
                                    multi_label=self.multi_label,
                                    binary=self.binary,
                                    labels=sorted(
                                        self.labels.values()),
                                    prefix='',
                                    logs=logs)
        y_val_t, y_pred_t = ClassificationMetric._transform_arrays(
            y_true=y_val,
            y_pred=y_pred,
            multi_label=self.multi_label,
            binary=self.binary)
        print(
            classification_report(y_val_t,
                                    y_pred_t,
                                    target_names=self.labels.keys()))
        if self.multi_label:
            print(
                multilabel_confusion_matrix(y_true=y_val_t,
                                            y_pred=y_pred_t,
                                            labels=sorted(
                                                self.labels.values())))
        else:
            print(
                confusion_matrix(y_true=np.argmax(y_val_t, axis=1) if
                                    len(y_val_t.shape) > 1 else y_val_t,
                                    y_pred=np.argmax(y_pred_t, axis=1) if
                                    len(y_pred_t.shape) > 1 else y_pred_t,
                                    labels=sorted(self.labels.values())))
        return

    def get_data(self):
        return self._data

    def compute_metrics(self,
                        y_val,
                        y_pred,
                        multi_label=False,
                        binary=False,
                        labels=None,
                        prefix='',
                        logs={}):
        all_classes_present = np.all(np.any(y_val > 0, axis=0))
        if multi_label or binary:
            binary_cutoffs = compute_binary_cutoffs(y_true=y_val,
                                                    y_pred=y_pred)
            self._binary_cutoffs.append(binary_cutoffs)
            print(f'Optimal cutoffs: {binary_cutoffs}')
        else:
            binary_cutoffs = None
        for i, cm in enumerate(CLASSIFICATION_METRICS):
            if all_classes_present or not (cm == ROC_AUC or cm == PR_AUC):
                metric = cm.compute(y_true=y_val,
                                    y_pred=y_pred,
                                    labels=labels,
                                    binary=binary,
                                    multi_label=multi_label,
                                    binary_cutoffs=binary_cutoffs)
                metric_value = metric.value
                print(f'{prefix} {cm.description}: {metric_value}')
                if not self._data:  # first recorded value
                    self._data.append({
                        f'{prefix}{KERAS_METRIC_QUANTITIES[cm]}':
                        metric_value,
                    })
                elif i == 0 and self._data and f'{prefix}{KERAS_METRIC_QUANTITIES[cm]}' in self._data[
                        -1].keys():
                    self._data.append({
                        f'{prefix}{KERAS_METRIC_QUANTITIES[cm]}':
                        metric_value,
                    })
                else:
                    self._data[-1][
                        f'{prefix}{KERAS_METRIC_QUANTITIES[cm]}'] = metric_value
                if len(
                        self._data
                ) > 1:  # this is the second epoch and metrics have been recorded for the first epoch
                    cur_best = self._data[-2][
                        f'{prefix}{KERAS_METRIC_QUANTITIES[cm]}_best']
                else:  # this is the first epoch
                    cur_best = metric_value

                new_best = metric_value if metric > cm(
                    value=cur_best) else cur_best

                self._data[-1][
                    f'{prefix}{KERAS_METRIC_QUANTITIES[cm]}_best'] = new_best

                logs[f'{prefix}{KERAS_METRIC_QUANTITIES[cm]}'] = metric_value
                logs[f'{prefix}{KERAS_METRIC_QUANTITIES[cm]}_best'] = new_best
            else:
                print(
                    f'Not all classes occur in the validation data, skipping ROC AUC and PR AUC.'
                )
        return logs


@dataclass(order=True)
class Metric(ABC):
    sort_index: float = field(init=False, repr=False)
    description: ClassVar[str] = 'Metric'
    key: ClassVar[str] = 'M'
    value: float
    scikit_scorer: ClassVar[_BaseScorer] = field(init=False, repr=False)
    greater_is_better: ClassVar[bool] = True

    def __post_init__(self):
        self.sort_index = self.value if self.greater_is_better else -self.value


@dataclass(order=True)
class ClassificationMetric(Metric, ABC):
    multi_label: bool = False
    binary: bool = False
    average: ClassVar[str] = None

    @classmethod
    @abstractmethod
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> Metric:
        pass

    @staticmethod
    def _transform_arrays(y_true: np.array,
                          y_pred: np.array,
                          multi_label: bool,
                          binary: bool,
                          binary_cutoffs: List[float] = None
                          ) -> (np.array, np.array):
        if binary:
            if len(y_pred.shape) > 1:
                y_pred = np.reshape(y_pred, -1)
            if len(y_true.shape) > 1:
                y_true = np.reshape(y_true, -1)
            assert (
                y_true.shape == y_pred.shape and len(y_true.shape) == 1
            ), f'Shapes of predictions and labels for binary classification should conform to (n_samples,) but received {y_pred.shape} and {y_true.shape}.'
            if binary_cutoffs is None:
                binary_cutoffs = compute_binary_cutoffs(y_true, y_pred)
            y_pred_transformed = np.where(y_pred > binary_cutoffs, 1, 0)
            y_true_transformed = y_true
        elif multi_label:
            assert (
                y_true.shape == y_pred.shape
            ), f'Shapes of predictions and labels for multilabel classification should conform to (n_samples, n_classes) but received {y_pred.shape} and {y_true.shape}.'
            if binary_cutoffs is None:
                binary_cutoffs = compute_binary_cutoffs(y_true, y_pred)
            y_pred_transformed = np.where(y_pred > binary_cutoffs, 1, 0)
            y_true_transformed = y_true
        else:
            if y_true.shape[1] > 1:
                y_true_transformed = np.zeros_like(y_true)
                y_true_transformed[range(len(y_true)), y_true.argmax(1)] = 1
            if y_pred.shape[1] > 1:
                y_pred_transformed = np.zeros_like(y_pred)
                y_pred_transformed[range(len(y_pred)), y_pred.argmax(1)] = 1
            assert (
                y_true.shape == y_pred.shape
            ), f'Shapes of predictions and labels for multiclass classification should conform to (n_samples,n_classes) but received {y_pred.shape} and {y_true.shape}.'
        return y_true_transformed, y_pred_transformed


@dataclass(order=True)
class MicroRecall(ClassificationMetric):
    description: ClassVar[str] = 'Micro Average Recall'
    average: ClassVar[str] = 'micro'
    key: ClassVar[str] = 'Recall/Micro'

    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(recall_score,
                                                       average='micro')
    greater_is_better: ClassVar[bool] = True

    @classmethod
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        y_true, y_pred = ClassificationMetric._transform_arrays(
            y_true=y_true,
            y_pred=y_pred,
            multi_label=multi_label,
            binary=binary,
            binary_cutoffs=binary_cutoffs)
        score = recall_score(y_true=y_true,
                             y_pred=y_pred,
                             labels=labels,
                             average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class UAR(MicroRecall):
    average: ClassVar[str] = 'macro'
    description: ClassVar[str] = 'Unweighted Average Recall'
    key: ClassVar[str] = 'Recall/Macro'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(recall_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True


@dataclass(order=True)
class Accuracy(ClassificationMetric):
    description: ClassVar[str] = 'Accuracy'
    key: ClassVar[str] = 'acc'

    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(accuracy_score)
    greater_is_better: ClassVar[bool] = True

    @classmethod
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        y_true, y_pred = ClassificationMetric._transform_arrays(
            y_true=y_true,
            y_pred=y_pred,
            multi_label=multi_label,
            binary=binary,
            binary_cutoffs=binary_cutoffs)
        score = accuracy_score(y_true=y_true, y_pred=y_pred)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class MacroF1(ClassificationMetric):
    average: ClassVar[str] = 'macro'
    description: ClassVar[str] = 'Macro Average F1'
    key: ClassVar[str] = 'F1/Macro'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(f1_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True

    @classmethod
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        y_true, y_pred = ClassificationMetric._transform_arrays(
            y_true=y_true,
            y_pred=y_pred,
            multi_label=multi_label,
            binary=binary,
            binary_cutoffs=binary_cutoffs)
        score = f1_score(y_true=y_true,
                         y_pred=y_pred,
                         labels=labels,
                         average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class MicroF1(MacroF1):
    average: ClassVar[str] = 'micro'
    description: ClassVar[str] = 'Micro Average F1'
    key: ClassVar[str] = 'F1/Micro'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(f1_score,
                                                       average='micro')
    greater_is_better: ClassVar[bool] = True


@dataclass(order=True)
class MacroPrecision(ClassificationMetric):
    average: ClassVar[str] = 'macro'
    description: ClassVar[str] = 'Macro Average Precision'
    key: ClassVar[str] = 'Prec/Macro'

    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(precision_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True

    @classmethod
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        y_true, y_pred = ClassificationMetric._transform_arrays(
            y_true=y_true,
            y_pred=y_pred,
            multi_label=multi_label,
            binary=binary,
            binary_cutoffs=binary_cutoffs)
        score = precision_score(y_true=y_true,
                                y_pred=y_pred,
                                labels=labels,
                                average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class MicroPrecision(MacroPrecision):
    average: ClassVar[str] = 'micro'
    description: ClassVar[str] = 'Micro Average Prec'
    key: ClassVar[str] = 'Prec/Micro'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(precision_score,
                                                       average='micro')
    greater_is_better: ClassVar[bool] = True


@dataclass(order=True)
class ROC_AUC(ClassificationMetric):
    average: ClassVar[str] = 'macro'
    description: ClassVar[
        str] = 'Area Under the Receiver Operating Characteristic Curve'
    key: ClassVar[str] = 'ROC AUC'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(roc_auc_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True

    @classmethod
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        score = roc_auc_score(y_true=y_true,
                              y_score=y_pred,
                              average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)


@dataclass(order=True)
class PR_AUC(ClassificationMetric):
    average: ClassVar[str] = 'macro'
    description: ClassVar[str] = 'Area Under the Precision Recall Curve'
    key: ClassVar[str] = 'PR AUC'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(average_precision_score,
                                                       average='macro')
    greater_is_better: ClassVar[bool] = True

    @classmethod
    def compute(cls,
                y_true: np.array,
                y_pred: np.array,
                labels: List,
                multi_label: bool,
                binary: bool,
                binary_cutoffs: List[float] = None) -> ClassificationMetric:
        score = average_precision_score(y_true=y_true,
                                        y_score=y_pred,
                                        average=cls.average)
        return cls(value=score, multi_label=multi_label, binary=binary)




def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


CLASSIFICATION_METRICS = all_subclasses(ClassificationMetric)


KERAS_METRIC_QUANTITIES = {
    M: f'val_{"_".join(M.key.lower().split(" "))}'
    for M in CLASSIFICATION_METRICS
}
KERAS_METRIC_MODES = {
    M: 'max' if M.greater_is_better else 'min'
    for M in CLASSIFICATION_METRICS
}

KEY_TO_METRIC = {metric.__name__: metric for metric in CLASSIFICATION_METRICS}
