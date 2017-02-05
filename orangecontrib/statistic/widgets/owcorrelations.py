from math import atan2, pi

import numpy
import scipy.stats
from AnyQt.QtCore import Qt
from PyQt4.QtCore import QSize, QTimer
from PyQt4.QtGui import QTableView, QStandardItemModel, QStandardItem, QColor

import Orange
from Orange.data import Table, ContinuousVariable, StringVariable
from Orange.widgets import gui
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph
from Orange.widgets.widget import OWWidget, AttributeList
from pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine


def pairwise_pearson_correlations(data, vars=None):
    '''
    :param data: table
    :param vars: changing names of columns for variables
    :return: matrix with values of pearson correlation
    Function counts half of values of pearson correlation
    and fills symetrically rest of matrix.
    '''
    if vars is None:
        vars = list(data.domain.variables)

    matrix_size = len(vars)
    matrix = numpy.zeros(shape=(matrix_size, matrix_size))

    all_vars = list(data.domain.variables)
    indices = [all_vars.index(v) for v in vars]

    for i, var_i in enumerate(indices):
        for j, var_j in enumerate(indices[i + 1:], i + 1):
            a = [row[var_i] for row in data]
            b = [row[var_j] for row in data]
            matrix[i, j] = scipy.stats.pearsonr(list(a), list(b))[0]

    matrix = matrix + matrix.T

    return matrix


def pairwise_spearman_correlations(data, vars=None):
    '''
    :param data: table
    :param vars: changing names of columns for variables
    :return: matrix with values of spearman correlation
    Function counts half of values of spearman correlation
    and fills symetrically rest of matrix.
    '''
    if vars is None:
        vars = list(data.domain.variables)

    matrix_size = len(vars)
    matrix = numpy.zeros(shape=(matrix_size, matrix_size))

    all_vars = list(data.domain.variables)
    indices = [all_vars.index(v) for v in vars]

    for i, var_i in enumerate(indices):
        for j, var_j in enumerate(indices[i + 1:], i + 1):
            a = [row[var_i] for row in data]
            b = [row[var_j] for row in data]
            matrix[i, j] = scipy.stats.spearmanr(list(a), list(b), 0)[0]
    matrix = matrix + matrix.T

    return matrix


def target_pearson_correlations(data, vars=None, target_var=None):
    '''
    :param data: table
    :param vars: changing names of columns for variables
    :param target_var: with this variable and all data is counting r value
    :return: list of r-value between target_var and column
    '''
    if vars is None:
        vars = list(data.domain.variables)

    if target_var is None:
        if data.domain.class_var.is_continuous:
            target_var = data.domain.class_var
        else:
            raise ValueError(
                "A data with continuous class variable expected if" +
                + "'target_var' is not explicitly declared.")

    all_vars = list(data.domain.variables)
    indices = [all_vars.index(v) for v in vars]
    target_index = all_vars.index(target_var)

    target_values = [row[target_index] for row in data]
    target_values = list(target_values)

    correlations = []
    for i, var_i in enumerate(indices):
        a = [row[var_i] for row in data]
        correlations.append(scipy.stats.pearsonr(list(a), target_values)[0])
    return correlations


def target_spearman_correlations(data, vars=None, target_var=None):
    '''
    :param data: table
    :param vars: changing names of columns for variables
    :param target_var: with this variable and all data is counting r value
    :return: list of r-value between target_var and column
    '''
    if vars is None:
        vars = list(data.domain.variables)

    if target_var is None:
        if data.domain.class_var.is_continuous:
            target_var = data.domain.class_var
        else:
            raise ValueError(
                "A data with continuous class variable expected if " +
                "'target_var' is not explicitly declared.")

    all_vars = list(data.domain.variables)
    indices = [all_vars.index(v) for v in vars]
    target_index = all_vars.index(target_var)

    target_values = [row[target_index] for row in data]
    target_values = list(target_values)

    correlations = []
    for i, var_i in enumerate(indices):
        a = [row[var_i] for row in data]
        correlations.append(scipy.stats.spearmanr(list(a),
                            target_values, 0)[0])
    return correlations


class OWCorrelations(OWWidget):
    name = "Correlations"

    description = "Calculate correlation"
    icon = "icons/correlation.svg"

    inputs = [("Data", Table, 'set_data')]
    outputs = [("Correlations", Table), ("Variables", AttributeList)]

    def __init__(self):
        super().__init__()
        self.data = None

        self.pairwise_correlations = True
        self.correlations_type = 0
        self.selected_index = None
        self.changed_flag = False
        self.auto_commit = True
        self.splitter_state = None
        self.corr_graph = CorrelationsGraph(self)
        self.mainArea.layout().addWidget(self.corr_graph.plot_widget)
        self.resize(1000, 500)  # TODO better size handling

        gui.radioButtonsInBox(
            self.controlArea, self, "correlations_type",
            ("Pairwise Pearson correlation",
             "Pairwise Spearman correlation"),
            box="Correlations",
            callback=self.on_corr_type_change
        )

        self.corr_table = CorrelationsTableView()

        self.corr_model = QStandardItemModel()
        self.corr_table.setModel(self.corr_model)

        self.controlArea.layout().addWidget(self.corr_table)
        self.corr_table.selectionModel().selectionChanged.connect(
             self.on_table_selection_change
         )

    @property
    def target_variable(self):
        if self.data:
            return self.data.domain.class_var
        else:
            return None

    def on_corr_type_change(self):
        """Do necessary actions after correlation type change.

        Clear computed data, set selected by user variables and finally
        commit(_if) changes.
        """
        if self.data is not None:
            curr_selection = self.selected_vars
            self.clear_computed()
            self.run()

            if curr_selection:
                try:
                    self.set_selected_vars(*curr_selection)
                except Exception as ex:
                    import traceback
                    traceback.print_exc()

            self.commit_if()

    def on_table_selection_change(self, selected, deselected):
        indexes = self.corr_table.selectionModel().selectedIndexes()
        if indexes:
            index = indexes[0]
            i, j = index.row(), index.column()
            if self.correlations_type == 2 and \
                    is_continuous(self.target_variable):
                j = len(self.var_names) - 1

            vars = [self.cont_vars[i], self.cont_vars[j]]
            self.corr_graph.update_data(vars[0], vars[1], i, j)
        else:
            vars = None
        self.selected_vars = vars

        self.send("Variables", vars)

    def clear_computed(self):
        """Clear computed data."""
        self.corr_model.clear()
        self.set_all_pairwise_matrix(None)
        self.set_target_correlations(None, None)

    def set_selected_vars(self, x, y):
        """Set selected by user variable(s)."""
        x = self.cont_vars.index(x)
        y = self.cont_vars.index(y)
        if self.correlations_type == 2:
            y = 0

        model = self.corr_model
        sel_model = self.corr_table.selectionModel()
        sel_model.select(model.index(x, y),
                         QItemSelectionModel.ClearAndSelect)

    def set_data(self, data):
        """
        Check if data has enough continuous variables.
        Update data, correlation type, correlation graph and commit changes.
        """
        self.clear()
        self.information()
        self.data = data
        if data is None:
            return
        if len(list(filter(lambda x: x.is_continuous, data.domain))) >= 2:
            self.set_variables_list(data)
            self.selected_index = None
            self.corr_graph.set_data(data)

            if self.selected_index is None or \
                    any(n in self.data.domain for n in self.selected_index):
                self.selected_index = self.var_names[:2]

            self.run()

        else:
            self.data = None
            self.information("Need data with at least 2 continuous variables.")

            self.commit_if()

        self.send("Correlations", Table(data))

    def clear(self):
        """ Clear all widget data. """
        self.data = None
        self.selected_vars = None
        self.clear_graph()

    def clear_graph(self):
        self.corr_graph._clear_plot_widget()
        self.corr_graph.set_data(None, None)
        self.corr_graph.replot()

    def set_variables_list(self, data):
        '''
        :param data: data
        :return: sets cont_vars and var_names
        '''
        vars = list(data.domain.variables)
        vars = [v for v in vars if v.is_continuous]
        self.cont_vars = vars
        self.var_names = [v.name for v in vars]

    def run(self):
        """ Start data matrix creation. """
        if self.correlations_type < 2:
            if self.correlations_type == 0:
                matrix = pairwise_pearson_correlations(self.data,
                                                       self.cont_vars)
            elif self.correlations_type == 1:
                matrix = pairwise_spearman_correlations(self.data,
                                                        self.cont_vars)
            self.set_all_pairwise_matrix(matrix)

        elif self.target_variable and self.target_variable.is_continuous:
            vars = [v for v in self.cont_vars if v != self.target_variable]
            p_corr = target_pearson_correlations(
                self.data, vars, self.target_variable)
            s_corr = target_spearman_correlations(
                self.data, vars, self.target_variable)
            correlations = [list(t) for t in zip(p_corr, s_corr)]
            self.set_target_correlations(correlations, vars)

    def set_all_pairwise_matrix(self, matrix):
        """ Set data matrix to correlations model and resize table. """
        self.matrix = matrix
        if matrix is not None:
            for i, row in enumerate(matrix):
                for j, e in enumerate(row):
                    item = QStandardItem()
                    if i != j:
                        item.setData(str(round(e, 5)), Qt.DisplayRole)
                    else:
                        item.setData(QColor(192, 192, 192), Qt.BackgroundRole)
                    self.corr_model.setItem(i, j, item)

            vars = self.cont_vars
            header = [v.name for v in vars]
            self.corr_model.setVerticalHeaderLabels(header)
            self.corr_model.setHorizontalHeaderLabels(header)

            self.corr_table.resizeColumnsToContents()
            self.corr_table.resizeRowsToContents()

            self.corr_table.updateGeometry()

    def set_target_correlations(self, correlations, vars=None):
        self.target_correlations = correlations
        if correlations is not None:
            for i, row in enumerate(correlations):
                for j, c in enumerate(row):
                    item = QStandardItem()
                    item.setData(c, Qt.DisplayRole)
                    self.corr_model.setItem(i, j, item)

            if vars is None:
                vars = self.cont_vars

            v_header = [v.name for v in vars]
            h_header = ["Pearson", "Spearman"]
            self.corr_model.setVerticalHeaderLabels(v_header)
            self.corr_model.setHorizontalHeaderLabels(h_header)

            self.corr_table.resizeColumnsToContents()
            self.corr_table.resizeRowsToContents()

            QTimer.singleShot(100, self.corr_table.updateGeometry)

    def commit_if(self):
        if self.auto_commit:
            self.commit()
        else:
            self.changed_flag = True

    def commit(self):
        table = None
        if self.data is not None:
            if self.correlations_type == 2 and self.target_variable and \
                    self.target_variable.is_continuous:
                pearson = ContinuousVariable.make("Pearson")
                spearman = ContinuousVariable.make("Spearman")
                row_name = StringVariable.make("Variable")

                domain = Orange.data.Domain([pearson, spearman],
                                            metas=[row_name])
                table = Orange.data.Table(domain, self.target_correlations)
                for inst, name in zip(table, self.var_names):
                    inst[row_name] = name
        self.send("Correlations", table)

    def selection_changed(self):  # TODO FIX IT
        pass


class CorrelationsTableView(QTableView):
    def sizeHint(self):
        return QSize(500, 300)  # TODO better size handling


class CorrelationsGraph(OWScatterPlotGraph):
    show_legend = False
    jitter_size = 10

    def __init__(self, scatter_widget, parent=None, _="None"):
        super().__init__(scatter_widget, parent, _)
        self.last_line = None

    def mouseMoveEvent(self, event):
        pass

    def update_data(self, x_attr, y_attr, i, j):
        """
        Update graph
        :param x_attr: x attributes
        :param y_attr: y attributes
        :return: None
        """
        OWScatterPlotGraph.update_data(self, x_attr, y_attr)

        x_index, y_index = i, j

        X = self.original_data[x_index]
        Y = self.original_data[y_index]

        valid = self.get_valid_list([x_index, y_index])

        X = X[valid]
        Y = Y[valid]
        x_min, x_max = self.attr_values[x_attr]

        X = numpy.array([numpy.ones_like(X), X]).T
        try:
            beta, _, _, _ = numpy.linalg.lstsq(X, Y)
        except numpy.linalg.LinAlgError:
            beta = [0, 0]

        angle = atan2(beta[1], 1) * 180/pi;
        ti = InfiniteLine((-999, beta[0] + -999 * beta[1]), angle=angle, pen='r')  # TODO FIX
        self.plot_widget.addItem(ti)
        if self.last_line is not None:
            self.plot_widget.removeItem(self.last_line)
        self.last_line = ti
        self.plot_widget.replot()
