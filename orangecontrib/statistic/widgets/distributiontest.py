from enum import Enum

import Orange.data
from AnyQt.QtCore import Qt
from Orange.widgets.utils import itemmodels
from Orange.widgets.widget import OWWidget, gui
from scipy.stats import shapiro, chisquare, anderson, kstest


class Distribution(Enum):
    """
    Data format: display_name, scipy_name
    """
    NORMAL = ('Normal', 'norm')
    UNIFORM = ('Uniform', None)
    OWN = ('Own', None)


class Test:
    allowed_distribution = NotImplemented
    name = NotImplemented

    @classmethod
    def compute(cls, widget):
        raise NotImplementedError


class KolmogorovSmirnov(Test):
    name = 'Kolmogorov-Smirnov'
    allowed_distribution = {Distribution.NORMAL}

    @classmethod
    def compute(cls, widget):
        if widget.distribution == Distribution.NORMAL:
            return widget.send('p-value', kstest(widget.column, 'norm').pvalue)


class AndersonDarling(Test):
    name = 'Anderson-Darling'
    allowed_distribution = {Distribution.NORMAL}

    @classmethod
    def compute(cls, widget):
        # FIXME: missing p-value
        return
        if widget.distribution == Distribution.NORMAL:
            return widget.send('p-value', anderson(widget.column).pvalue)


class ShapiroWilk(Test):
    name = 'Shapiro-Wilk'
    allowed_distribution = {Distribution.NORMAL}

    @classmethod
    def compute(cls, widget):
        if widget.distribution == Distribution.NORMAL:
            return widget.send('p-value', shapiro(widget.column)[1])


class ChiSquare(Test):
    name = 'Chi-square'
    allowed_distribution = {Distribution.NORMAL}

    @classmethod
    def compute(cls, widget):
        if widget.distribution == Distribution.UNIFORM:
            return widget.send('p-value', chisquare(widget.column).pvalue[0])


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
    available_distributions = (
        Distribution.NORMAL,
        Distribution.UNIFORM,
        Distribution.OWN,
    )
    test_idx = 0
    distribution_idx = 0
    column_idx = 0
    own_distribution_idx = -1

    def __init__(self):
        super().__init__()

        box = gui.vBox(self.controlArea, 'Tests')
        gui.radioButtonsInBox(
            box, self, 'test_idx',
            btnLabels=[test.name for test in self.available_tests],
            callback=self.test_changed,
        )

        box = gui.vBox(self.controlArea, 'Distributions')
        gui.radioButtonsInBox(
            box, self, 'distribution_idx',
            btnLabels=[distribution.value[0]
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

    def set_data(self, data):
        if data is not None:
            self.data = data
            self.available_columns[:] = data.domain

    def test_changed(self):
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
            self.test.compute(self)

    @property
    def test(self):
        return self.available_tests[self.test_idx]

    @property
    def distribution(self):
        return self.available_distributions[self.distribution_idx]

    @property
    def column(self):
        return self.data[:, self.column_idx].X

    @property
    def own_distribution(self):
        return self.data[:, self.own_distribution_idx].X
