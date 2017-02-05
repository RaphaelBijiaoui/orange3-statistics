import math
import sys
from itertools import chain

import Orange
import Orange.data
import Orange.data
import Orange.data
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget
from matplotlib.backends.backend_qt4agg import \
    FigureCanvasQTAgg as FigureCanvas
from statsmodels.graphics.gofplots import qqplot_2samples


class ProbabilityPlot(OWWidget):
    name = "Probability Plot"
    icon = "icons/mywidget.svg"
    want_main_area = True
    inputs = [("Data", Orange.data.Table, "set_data")]
    settingsHandler = DomainContextHandler()
    attribute = ContextSetting(None)
    group_var = ContextSetting(None)

    def __init__(self):
        super().__init__()
        self.distribution_idx = 0
        self.var_data = np.array([])
        self.column_data = np.array([])
        self.dataset = None
        self.column_idx = 0
        self.var_idx = 0
        self.available_plot = ["Probability Plot", "Q-Q Plot", "P-P Plot",
                               "Q-Q Plot of 2 samples"]
        self.attrs = VariableListModel()
        self.all_attrs = VariableListModel()

        view = gui.listView(
            self.controlArea, self, "attribute", box="First variable",
            model=self.attrs, callback=self.attr_changed)
        self.view2 = gui.listView(
            self.controlArea, self, "group_var", box="Second variable",
            model=self.attrs, callback=self.var_changed)
        box = gui.vBox(self.controlArea, 'Type of plot')
        self.distribution_choose = gui.radioButtonsInBox(
            box, self, 'distribution_idx',
            btnLabels=self.available_plot,
            callback=self.plot_changed,
        )
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.mainArea.frameGeometry().width()
        self.mainArea.layout().addWidget(self.canvas)

    def set_data(self, dataset):
        self.view2.hide()
        self.clear_plot()
        if dataset is not None and (
                    not bool(dataset) or not len(dataset.domain)):
            dataset = None
        self.closeContext()
        self.dataset = dataset
        self.attribute = None
        if dataset:
            domain = dataset.domain

            # all atributes from dataset
            self.all_attrs[:] = list(domain) + \
                                [meta for meta in domain.metas
                                 if meta.is_continuous or meta.is_discrete]
            # atributes in list
            self.attrs[:] = [a for a in chain(domain.variables, domain.metas)
                             if a.is_continuous]
            # initial
            if self.attrs:
                self.attribute = self.attrs[0]
                self.group_var = self.attrs[0]
                self.openContext(self.dataset)
                self.var_changed()
                self.attr_changed()

    def plot_changed(self):
        '''
        Selection of type of plot.
        :return: plot function
        '''
        self.clear_plot()

        if self.distribution_idx == 1:
            self.view2.hide()
            self.qq_plot()
        if self.distribution_idx == 2:
            self.view2.hide()
            self.pp_plot()
        if self.distribution_idx == 3:
            self.view2.show()
            self.qq_plot_2samples()
        if self.distribution_idx == 0:
            self.view2.hide()
            self.prob_plot()

    def prob_plot(self):
        '''

        :return:  Probability plot
        '''
        self.ax = self.figure.add_subplot(111)
        self.ax.hold(True)
        stats.probplot(self.column_data, dist="norm", plot=plt)
        self.canvas.draw()

    def qq_plot_2samples(self):
        '''

        :return: Q-Q plot between two samples
        '''
        self.ax = self.figure.add_subplot(111)
        self.ax.hold(True)
        pp_x = sm.ProbPlot(self.column_data)
        pp_y = sm.ProbPlot(self.var_data)
        qqplot_2samples(pp_x, pp_y, ax=self.ax)
        self.canvas.draw()

    def pp_plot(self):
        '''

        :return: P-P plot
        '''
        self.ax = self.figure.add_subplot(111)
        self.ax.hold(True)
        probplot = sm.ProbPlot(self.column_data)
        probplot.ppplot(ax=self.ax, line='45')
        self.canvas.draw()

    def qq_plot(self):
        '''

        :return: Q-Q plot
        '''
        self.ax = self.figure.add_subplot(111)
        self.ax.hold(True)
        sm.qqplot(self.column_data, line="q", ax=self.ax)
        self.canvas.draw()

    def clear_plot(self):
        '''
        After all change of type of plot or one of atributes
        :return: clear plot - blank
        '''
        self.ax = self.figure.add_subplot(111)
        self.ax.hold(False)
        self.ax.plot([], '*-')
        self.canvas.draw()

    def attr_changed(self):
        '''
        Select index of column.
        :return: change plot
        '''
        self.clear_plot()
        for i in enumerate(self.all_attrs):
            if self.attribute == i[1]:
                self.column_idx = i[0]
        self.var_data = self.var()
        self.column_data = self.column()
        self.plot_changed()

    def var_changed(self):
        '''
        Select index of secound column to 2 samples to qq plot.
        :return: change plot
        '''
        self.clear_plot()
        for i in enumerate(self.all_attrs):
            if self.group_var == i[1]:
                self.var_idx = i[0]
        self.var_data = self.var()
        self.column_data = self.column()
        self.plot_changed()

    def column(self):
        '''
        Chose data and set 0.0 in missing data
        :return: data of choosen column
        '''
        l = self.dataset[:, self.column_idx]
        result = []
        for sublist in l:
            for item in sublist:
                if math.isnan(item):
                    result.append(0.0)
                else:
                    result.append(item)
        return np.array(result)

    def var(self):
        '''
        Chose data and set 0.0 in missing data
        :return: data of choosen second column
        '''
        l = self.dataset[:, self.var_idx]
        result = []
        for sublist in l:
            for item in sublist:
                if math.isnan(item):
                    result.append(0.0)
                else:
                    result.append(item)
        return np.array(result)

    def clear_scene(self):
        self.closeContext()
        self.openContext(self.dataset)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "heart_disease"

    data = Orange.data.Table(filename)
    w = ProbabilityPlot()
    w.show()
    w.raise_()
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.saveSettings()
    return rval


if __name__ == "__main__":
    sys.exit(main())
