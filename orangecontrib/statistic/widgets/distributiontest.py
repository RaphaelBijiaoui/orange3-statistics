from enum import Enum

import Orange.data
from AnyQt.QtCore import Qt
from Orange.widgets.utils import itemmodels
from Orange.widgets.widget import OWWidget, gui
from scipy.stats import shapiro, chisquare, anderson, kstest


class Distribution(Enum):
    NORMAL = 'Normal'
    UNIFORM = 'Uniform'
    EXPONENTIAL = 'Exponential'
    LOGISTIC = 'Logistic'
    OWN = 'Own'


def distribution_to_numpy(distribution: Distribution) -> str:
    if distribution == Distribution.NORMAL:
        return 'norm'
    elif distribution == Distribution.EXPONENTIAL:
        return 'expon'
    elif distribution == Distribution.LOGISTIC:
        return 'logistic'


class Test:
    allowed_distribution = NotImplemented
    name = NotImplemented

    @classmethod
    def compute(cls, widget):
        raise NotImplementedError


class KolmogorovSmirnov(Test):
    name = 'Kolmogorov-Smirnov'
    allowed_distribution = {
        Distribution.NORMAL,
        Distribution.EXPONENTIAL,
        Distribution.LOGISTIC,
    }

    @classmethod
    def compute(cls, widget):
        np_dist = distribution_to_numpy(widget.distribution)
        return kstest(widget.column, np_dist).pvalue


class AndersonDarling(Test):
    name = 'Anderson-Darling'
    allowed_distribution = {Distribution.NORMAL}

    @classmethod
    def compute(cls, widget):
        # FIXME: missing p-value
        return
        return anderson(widget.column).pvalue


class ShapiroWilk(Test):
    name = 'Shapiro-Wilk'
    allowed_distribution = {Distribution.NORMAL}

    @classmethod
    def compute(cls, widget):
        if widget.distribution == Distribution.NORMAL:
            return shapiro(widget.column)[1]


class ChiSquare(Test):
    name = 'Chi-square'
    allowed_distribution = {Distribution.UNIFORM}

    @classmethod
    def compute(cls, widget):
        if widget.distribution == Distribution.UNIFORM:
            return chisquare(widget.column).pvalue[0]


class DistributionTest(OWWidget):
    name = 'Distribution Test'
    description = 'Check if data is in given distribution.'
    icon = 'icons/mywidget.svg'
    want_main_area = False
    buttons_area_orientation = Qt.Vertical
    resizing_enabled = False
    inputs = [('Data', Orange.data.Table, 'set_data')]
    outputs = [('p-value', float)]

    available_tests = (
        KolmogorovSmirnov,
        AndersonDarling,
        ShapiroWilk,
        ChiSquare,
    )
    available_distributions = [d for d in Distribution]
    test_idx = 0
    distribution_idx = 0
    column_idx = 0
    own_distribution_idx = 0

    def __init__(self):
        super().__init__()

        box = gui.vBox(self.controlArea, 'Tests')
        gui.radioButtonsInBox(
            box, self, 'test_idx',
            btnLabels=[test.name for test in self.available_tests],
            callback=self.test_changed,
        )

        box = gui.vBox(self.controlArea, 'Distributions')
        self.distribution_choose = gui.radioButtonsInBox(
            box, self, 'distribution_idx',
            btnLabels=[distribution.value
                       for distribution
                       in self.available_distributions],
            callback=self.distribution_changed,
        )

        self.column_chose = gui.comboBox(
            self.controlArea, self, 'column_idx',
            box='Selected column',
            items=[],
            orientation=Qt.Horizontal,
            callback=self.column_changed,
        )
        self.available_columns = itemmodels.VariableListModel(parent=self)
        self.column_chose.setModel(self.available_columns)

        self.own_distribution_choose = gui.comboBox(
            self.controlArea, self, 'own_distribution_idx',
            box='Own distribution',
            items=[],
            orientation=Qt.Horizontal,
            callback=self.column_changed,
            disabled=True
        )
        self.own_distribution_choose.setModel(self.available_columns)
        self.data = None
        self.test_changed()

    def set_data(self, data):
        if data is not None:
            self.data = data
            self.available_columns[:] = data.domain

    def test_changed(self):
        for idx, button in enumerate(self.distribution_choose.buttons):
            if Distribution(button.text()) in self.test.allowed_distribution:
                button.setDisabled(False)
                if self.distribution not in self.test.allowed_distribution:
                    button.toggle()
                    self.distribution_idx = idx
            else:
                button.setDisabled(True)
        return self.compute_p_value()

    def distribution_changed(self):
        if self.distribution == Distribution.OWN:
            self.own_distribution_choose.setDisabled(False)
        else:
            self.own_distribution_choose.setDisabled(True)
        return self.compute_p_value()

    def column_changed(self):
        return self.compute_p_value()

    def compute_p_value(self):
        if self.data is not None:
            p_value = self.test.compute(self)
            return self.send('p-value', p_value)

    @property
    def test(self) -> Test:
        return self.available_tests[self.test_idx]

    @property
    def distribution(self) -> Distribution:
        return self.available_distributions[self.distribution_idx]

    @property
    def column(self):
        return self.data[:, self.column_idx].X

    @property
    def own_distribution(self):
        return self.data[:, self.own_distribution_idx].X
