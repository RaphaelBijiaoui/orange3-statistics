from enum import Enum

import Orange.data
from AnyQt.QtCore import Qt
from Orange.widgets.utils import itemmodels
from Orange.widgets.widget import OWWidget, gui
from scipy.stats import shapiro, chisquare, anderson, kstest


class Test(Enum):
    KOLMOGOROV_SMIRNOV = 'Kolmogorov-Smirnov'
    ANDERSON_DARLING = 'Anderson-Darling'
    SHAPIRO_WILK = 'Shapiro-Wilk'
    CHI_SQUARE = 'Chi-square'


class Distribution(Enum):
    """
    Data format: display_name, scipy_name
    """
    NORMAL = ('Normal', 'norm')
    UNIFORM = ('Uniform', None)


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
        Test.KOLMOGOROV_SMIRNOV,
        Test.ANDERSON_DARLING,
        Test.SHAPIRO_WILK,
        Test.CHI_SQUARE,
    )
    chosen_test = 0
    available_distributions = (
        Distribution.NORMAL,
        Distribution.UNIFORM,
    )
    chosen_distribution = 0
    chosen_column = 0

    def __init__(self):
        super().__init__()

        box = gui.vBox(self.controlArea, 'Tests')
        gui.radioButtonsInBox(
            box,
            self,
            'chosen_test',
            btnLabels=[test.value for test in self.available_tests],
            callback=self.test_changed,
        )

        box = gui.vBox(self.controlArea, 'Distributions')
        gui.radioButtonsInBox(
            box,
            self,
            'chosen_distribution',
            btnLabels=[distribution.value[0]
                       for distribution
                       in self.available_distributions],
            callback=self.distribution_changed,
        )

        self.column_chose = gui.comboBox(
            self.controlArea, self, 'chosen_column', box='Selected column',
            items=[],
            orientation=Qt.Horizontal, callback=self.column_changed)
        self.available_columns = itemmodels.VariableListModel(parent=self)
        self.column_chose.setModel(self.available_columns)
        self.data = [1, 2, 3]

    def set_data(self, data):
        if data is not None:
            self.data = data
            self.available_columns[:] = data.domain

    def test_changed(self):
        return self.compute_p_value()

    def distribution_changed(self):
        return self.compute_p_value()

    def column_changed(self):
        return self.compute_p_value()

    def compute_p_value(self):
        if self.data is None:
            return
        if self.test == Test.KOLMOGOROV_SMIRNOV:
            return self.kolmogorov_smirnov()
        elif self.test == Test.ANDERSON_DARLING:
            return self.anderson_darling()
        elif self.test == Test.SHAPIRO_WILK:
            return self.shapiro_wilk()
        elif self.test == Test.CHI_SQUARE:
            return self.chi_square()

    @property
    def test(self):
        return self.available_tests[self.chosen_test]

    @property
    def distribution(self):
        return self.available_distributions[self.chosen_distribution]

    @property
    def column(self):
        return self.data[:, self.chosen_column].X

    def kolmogorov_smirnov(self):
        if self.distribution == Distribution.NORMAL:
            return self.send('p-value', kstest(self.column, 'norm').pvalue)

    def anderson_darling(self):
        # FIXME: missing p-value
        return
        if self.distribution == Distribution.NORMAL:
            return self.send('p-value', anderson(self.column).pvalue)

    def shapiro_wilk(self):
        if self.distribution == Distribution.NORMAL:
            return self.send('p-value', shapiro(self.column)[1])

    def chi_square(self):
        if self.distribution == Distribution.UNIFORM:
            return self.send('p-value', chisquare(self.column).pvalue)
