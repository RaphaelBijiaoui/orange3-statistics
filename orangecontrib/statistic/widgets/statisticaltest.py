from abc import abstractmethod

import Orange.data
from statsmodels.stats.weightstats import ztest
from AnyQt.QtWidgets import QHBoxLayout, QListView
from Orange.widgets.utils import itemmodels
from Orange.widgets.widget import OWWidget, gui
from scipy.stats import ttest_1samp, ttest_ind, fisher_exact, f_oneway


class StatisticalTestWidget(OWWidget):
    name = 'Statistical test'
    description = 'Do statistical tests and returns p-value'
    icon = 'icons/test_widget.svg'
    want_main_area = False
    inputs = [('Data', Orange.data.Table, 'set_data')]
    outputs = [('p-value', float)]

    def __init__(self):
        super().__init__()
        self.data = None
        self.available_tests = [TTest(), ZTest(), FisherTest(), Anova()]
        self.active_tests = []

        self.available_columns = itemmodels.VariableListModel(parent=self)

        vlayout = QHBoxLayout()
        # Data selection
        gui.widgetBox(
            self.controlArea, self.tr('Chose data to test'),
            orientation=vlayout, spacing=16
        )
        self.varview = QListView(selectionMode=QListView.MultiSelection)
        self.varview.setModel(self.available_columns)
        self.varview.selectionModel().selectionChanged.connect(
            self.update_column_selection
        )
        vlayout.addWidget(self.varview)

        # Test selection

        self.tests = itemmodels.VariableListModel(parent=self)
        self.testview = QListView(selectionMode=QListView.SingleSelection)
        self.testview.setModel(self.tests)
        self.testview.selectionModel().selectionChanged.connect(
            self.test_selected
        )
        vlayout.addWidget(self.testview)

    def update_column_selection(self, *args):
        columns_index = self.selected_columns
        if len(columns_index) == 1:
            self.enable_test_with_one_sample()
        elif len(columns_index) == 2:
            self.enable_tests_with_two_samples()
        elif len(columns_index) >= 2:
            self.enable_tests_with_many_samples()

    def enable_test_with_one_sample(self):
        self.active_tests = [test for test in self.available_tests if
                             test.has_one_sample]
        self.tests[:] = [test.name for test in self.active_tests]

    def enable_tests_with_two_samples(self):
        self.active_tests = [test for test in self.available_tests if
                             test.has_two_sample]
        self.tests[:] = [test.name for test in self.active_tests]

    def enable_tests_with_many_samples(self):
        self.active_tests = [test for test in self.available_tests if
                             test.has_many_sample]
        self.tests[:] = [test.name for test in self.active_tests]

    @property
    def selected_columns(self):
        rows = self.varview.selectionModel().selectedRows()
        return [index.row() for index in rows]

    @property
    def selected_test(self):
        # FIXME: I have no idea why test list selection sometimes disappearing
        try:
            _ = self.testview.selectionModel().selectedRows()[0].row()
            self.last_selected_test = _
        except IndexError:
            pass
        return self.last_selected_test

    def set_data(self, data):
        return self.update_data(data)

    def update_data(self, data):
        if data:
            self.data = data
            self.available_columns[:] = data.domain

    def column_changed(self):
        pass

    def test_selected(self):
        return self.do_test()

    def do_test(self):
        if not self.data:
            return
        try:
            test = self.active_tests[self.selected_test]
        except IndexError:
            return
        columns_indexes = self.selected_columns
        number_of_columns = len(columns_indexes)
        if number_of_columns == 0:
            return
        elif number_of_columns == 1:
            p_value = test.one_sample_test(
                self.data[:, columns_indexes].X)
        elif number_of_columns == 2:
            p_value = test.two_sample_test(
                *[self.data[:, column_index].X for column_index in
                  columns_indexes])
        else:
            p_value = test.many_sample_test(
                *[self.data[:, column_index].X for column_index in
                  columns_indexes])
        return self.send('p-value', p_value)


class StatisticalTest:
    name = ''
    has_one_sample = False
    has_two_sample = False
    has_many_sample = False

    @abstractmethod
    def one_sample_test(self, sample) -> float:
        raise NotImplementedError

    @abstractmethod
    def two_sample_test(self, sample_1, sample_2) -> float:
        raise NotImplementedError

    @abstractmethod
    def many_sample_test(self, *samples) -> float:
        raise NotImplementedError


class TTest(StatisticalTest):
    name = 'T-Test'

    has_one_sample = True
    has_two_sample = True

    def __init__(self):
        self.excepted_value = 0

    def one_sample_test(self, sample) -> float:
        return ttest_1samp(sample, self.excepted_value).pvalue[0]

    def two_sample_test(self, sample_1, sample_2) -> float:
        return ttest_ind(sample_1, sample_2).pvalue[0]


class ZTest(StatisticalTest):
    name = 'Z-Test'

    has_one_sample = True
    has_two_sample = True

    def one_sample_test(self, sample) -> float:
        return ztest(sample)[1][0]

    def two_sample_test(self, sample_1, sample_2) -> float:
        return ztest(sample_1, sample_2)[1][0]


class FisherTest(StatisticalTest):
    name = 'Fisher Test'
    has_two_sample = True

    def two_sample_test(self, sample_1, sample_2) -> float:
        values_1 = set(value[0] for value in sample_1)
        values_2 = set(value[0] for value in sample_2)
        if len(values_1) == 2 and len(values_2) == 2:
            values = list(values_1)
            data_1 = {v: 0 for v in values}
            data_2 = {v: 0 for v in values}
            for value in sample_1:
                data_1[value[0]] += 1
            for value in sample_2:
                data_2[value[0]] += 1
            return fisher_exact(
                [
                    [data_1[v] for v in values],
                    [data_2[v] for v in values]
                ]
            )[1]


class Anova(StatisticalTest):
    name = 'Anova (one way)'

    has_many_sample = True

    def many_sample_test(self, *samples) -> float:
        return f_oneway(*samples)[1]
