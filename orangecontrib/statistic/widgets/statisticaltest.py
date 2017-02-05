from abc import abstractmethod

import Orange.data
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QHBoxLayout, QListView
from Orange.widgets.utils import itemmodels
from Orange.widgets.widget import OWWidget, gui
from scipy.stats import ttest_1samp, ttest_ind, fisher_exact, f_oneway
from statsmodels.stats.weightstats import ztest


class StatisticalTestWidget(OWWidget):
    """
    Perform statistical tests
    """
    name = 'Statistical test'
    description = 'Do statistical tests and returns p-value.\n' \
                  'Many samples should be in same table as columns.'
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

        self.available_corrections = [
            None, BonferroniCorrection(), SidakCorrection()]

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

        # Correction selection
        self.corrections = itemmodels.VariableListModel(parent=self)
        self.corrections[:] = \
            [str(None), BonferroniCorrection().name, SidakCorrection().name]
        self.cor_varview = QListView(selectionMode=QListView.SingleSelection)
        self.cor_varview.setModel(self.corrections)
        self.cor_varview.selectionModel().selectionChanged.connect(
            self.set_correction
        )
        vlayout.addWidget(self.cor_varview)

        self.n_of_tests = 1

        self.n_of_tests_input = gui.lineEdit(
            self.controlArea, self,
            'n_of_tests',
            label='<p align="right">Number of tests</p>',
            callbackOnType=True,
            controlWidth=150,
            orientation=Qt.Horizontal,
            callback=self.number_of_tests_changed,
        )

        self.chosen_correction = None

        pval_box = gui.vBox(self.controlArea)
        self.pval_infolabel = gui.widgetLabel(
            pval_box,
            '<p align="left"><b>p-value: </b></p>',
        )

    def show_p_value(self, p_value: float):
        """
        Draw p-value in widget if only is numeric.
        """
        if isinstance(p_value, int) or isinstance(p_value, float):
            p_val_str = str(p_value)
        else:
            p_val_str = '~ ? ~'
        self.pval_infolabel.setText(
            '<p align="left"><b>p-value: {}</b></p>'.format(p_val_str))

    def update_column_selection(self, *args):
        """
        Method called when selection of samples changed.
        Enables matching tests for selected samples.
        """
        columns_index = self.selected_columns
        self.active_tests = []
        self.tests[:] = []
        if (not self.selected_data_is_continuous and
                not self.selected_data_is_discrete):
            return
        elif len(columns_index) == 1:
            self.enable_test_with_one_sample()
        elif len(columns_index) == 2:
            self.enable_tests_with_two_samples()
        elif len(columns_index) >= 2:
            self.enable_tests_with_many_samples()

    def enable_test_with_one_sample(self):
        """
        Enable tests which can be apply to one sample.
        """
        self.active_tests = [test for test in self.available_tests if
                             test.can_be_used_with_one_sample(self)]
        self.tests[:] = [test.name for test in self.active_tests]

    def enable_tests_with_two_samples(self):
        """
        Enable tests which can be apply to two samples.
        """
        self.active_tests = [test for test in self.available_tests if
                             test.can_be_used_with_two_sample(self)]
        self.tests[:] = [test.name for test in self.active_tests]

    def enable_tests_with_many_samples(self):
        """
        Enable tests which can be apply to many samples.
        It should not be called for one or two data samples!
        """
        self.active_tests = [test for test in self.available_tests if
                             test.can_be_used_with_many_sample(self)]
        self.tests[:] = [test.name for test in self.active_tests]

    @property
    def selected_columns(self):
        """
        :return: Indexes of actually selected columns.
        """
        rows = self.varview.selectionModel().selectedRows()
        return [index.row() for index in rows]

    @property
    def selected_data_is_continuous(self):
        """
        :return: Tell whether all selected samples are continuous.
        """
        return all([self.available_columns[i].is_continuous for i in
                    self.selected_columns])

    @property
    def selected_data_is_discrete(self):
        """
        :return: Tell whether all selected samples are discrete.
        """
        return all([not self.available_columns[i].is_continuous for i in
                    self.selected_columns])

    @property
    def selected_test(self):
        """
        :return: index of actually selected test.
        """
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
        """
        Method called when input data updated.
        :param data: input data
        """
        if data:
            self.data = data
            self.available_columns[:] = data.domain

    def test_selected(self):
        """
        Method called when test selected.
        """
        return self.do_test()

    def do_test(self):
        """
        Main method of widget. Do all test logic.
        Call it if you want to do test or recalculate it.
        """
        if not self.data:
            return
        try:
            test = self.active_tests[self.selected_test]
        except IndexError:
            return
        except AttributeError:
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
        if self.chosen_correction:
            chosen_ = self.available_corrections[self.chosen_correction]
            p_value = chosen_.calculate_correction(p_value, self.n_of_tests)
        self.show_p_value(p_value)
        return self.send('p-value', p_value)

    def number_of_tests_changed(self):
        """
        Method called when number of test in correction changed.
        """
        try:
            self.n_of_tests = int(self.n_of_tests)
        except TypeError:
            self.n_of_tests = 1
            self.n_of_tests_input = "1"
        self.test_selected()

    def set_correction(self):
        """
        Method called when correction method changed.
        """
        new_cor = self.cor_varview.selectionModel().selectedRows()[0].row()
        recalculate = False
        if self.chosen_correction != new_cor:
            self.chosen_correction = new_cor
            recalculate = True
        if recalculate:
            self.test_selected()


class Correction:
    """
    Base Correction class, all Corrections should be subclass of this
    """
    name = ''

    @abstractmethod
    def calculate_correction(self, p_value: float, n_of_tests: int) -> float:
        """
        :param p_value: p-value from statistical test
        :param n_of_tests: number of tests
        :return:
        """
        raise NotImplementedError


class BonferroniCorrection(Correction):
    name = 'Bonferroni'

    def calculate_correction(self, p_value: float, n_of_tests: int) -> float:
        return p_value * n_of_tests


class SidakCorrection(Correction):
    name = 'Sidak'

    def calculate_correction(self, p_value: float, n_of_tests: int) -> float:
        return 1 - (1 - p_value) ** n_of_tests


class StatisticalTest:
    """
    Base, abstract class for all statistical tests.
    """
    name = ''
    has_one_sample = False
    has_two_sample = False
    has_many_sample = False
    use_continuous_data = False
    use_discrete_data = False

    @abstractmethod
    def one_sample_test(self, sample) -> float:
        """
        Do test using one sample
        :param sample: data to test
        :return: p-value
        """
        raise NotImplementedError

    @abstractmethod
    def two_sample_test(self, sample_1, sample_2) -> float:
        """
        Do test using two samples
        :param sample_1: first sample of data
        :param sample_2: second sample of data
        :return: p-value
        """
        raise NotImplementedError

    @abstractmethod
    def many_sample_test(self, *samples) -> float:
        """
        :param samples: As many data samples as you want.
        :return: p-value
        """
        raise NotImplementedError

    def can_be_used_with_one_sample(self, widget):
        """
        Tells if this test can be done using actually selected data in widget
        """
        return self.has_one_sample and self.can_be_used_in(widget)

    def can_be_used_with_two_sample(self, widget: StatisticalTestWidget):
        """
        Tells if this test can be done using actually selected data in widget
        """
        return self.has_two_sample and self.can_be_used_in(widget)

    def can_be_used_with_many_sample(
            self, widget: StatisticalTestWidget) -> bool:
        """
        Tells if this test can be done using actually selected data in widget
        """
        return self.has_many_sample and self.can_be_used_in(widget)

    def can_be_used_in(self, widget) -> bool:
        """
        Tells if this test can be done using actually selected data in widget.
        Check only if data is continuous or not.
        """
        if widget.selected_data_is_continuous and self.use_continuous_data:
            return True
        elif widget.selected_data_is_discrete and self.use_discrete_data:
            return True
        else:
            return False


class TTest(StatisticalTest):
    name = 'T-Test'

    has_one_sample = True
    has_two_sample = True

    use_continuous_data = True

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

    use_continuous_data = True

    def one_sample_test(self, sample) -> float:
        return ztest(sample)[1][0]

    def two_sample_test(self, sample_1, sample_2) -> float:
        return ztest(sample_1, sample_2)[1][0]


class FisherTest(StatisticalTest):
    name = 'Fisher Test'

    has_two_sample = True

    use_discrete_data = True

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

    use_continuous_data = True

    def many_sample_test(self, *samples) -> float:
        return f_oneway(*samples)[1][0]
