import collections
import sys
from enum import Enum
from math import sqrt
from xml.sax.saxutils import escape
import Orange
import Orange.data
import numpy as np
import pyqtgraph as pg
import statsmodels.stats.diagnostic as ande
from AnyQt.QtCore import QRectF
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor, QBrush, QPainter, QPicture
from AnyQt.QtGui import QPen, QPalette
from AnyQt.QtWidgets import QSizePolicy, QLabel
from Orange.statistics import distribution, contingency
from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.visualize.owlinearprojection import LegendItem, \
    ScatterPlotItem
from Orange.widgets.widget import OWWidget
from PyQt4.QtGui import QApplication
from scipy.stats import shapiro, chisquare, kstest, ks_2samp


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
    elif distribution == Distribution.OWN:
        return 'own'


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
        Distribution.OWN
    }

    @classmethod
    def compute(cls, widget):
        """
        Kolmogorov-Smirnov test for one or two samples from initial data
        :param widget:
        :return: p-value of Kolmogorov-Smirnov test
        """
        np_dist = distribution_to_numpy(widget.distribution)
        if np_dist != 'own':

            return kstest(widget.column, np_dist).pvalue
        else:
            try:
                columns = [a[0] for a in widget.column]
                own = [a[0] for a in widget.own_distribution]
            except:
                # FIXME:only for continuos samples
                return 0
            return ks_2samp(columns, own).pvalue


class AndersonDarling(Test):
    name = 'Anderson-Darling'
    allowed_distribution = {Distribution.NORMAL}

    @classmethod
    def compute(cls, widget):
        """
        Anderson-Darling test for one sample
        :param widget:
        :return: p-value for Anderson-Darling test
        """
        if isinstance(
                ande.normal_ad(np.array([a[0] for a in widget.column]))[1],
                float):

            return ande.normal_ad(np.array([a[0] for a in widget.column]))[1]
        else:
            return ande.normal_ad(np.array([a[0] for a in widget.column]))[1][
                0]


class ShapiroWilk(Test):
    name = 'Shapiro-Wilk'
    allowed_distribution = {Distribution.NORMAL}

    @classmethod
    def compute(cls, widget):
        """
        Shapiro-Wilk test for one sample.
        :param widget:
        :return: p-value for Shapiro-Wilk test
        """
        if widget.distribution == Distribution.NORMAL:
            return shapiro(widget.column)[1]


class ChiSquare(Test):
    name = 'Chi-square'
    allowed_distribution = {Distribution.UNIFORM}

    @classmethod
    def compute(cls, widget):
        """
        Chi-square test for one sample
        :param widget:
        :return: p-value of Chi-square test
        """
        if widget.distribution == Distribution.UNIFORM:
            return chisquare(widget.column).pvalue[0]


class DistributionBarItem(pg.GraphicsObject):
    """
    Overwrite Class to paint appopriate bars
    """

    def __init__(self, geometry, dist, colors):
        super().__init__()
        self.geometry = geometry
        self.dist = dist
        self.colors = colors
        self.__picture = None

    def paint(self, painter, options, widget):
        if self.__picture is None:
            self.__paint()
        painter.drawPicture(0, 0, self.__picture)

    def boundingRect(self):
        return self.geometry

    def __paint(self):
        picture = QPicture()
        painter = QPainter(picture)
        pen = QPen(QBrush(Qt.white), 0.5)
        pen.setCosmetic(True)
        painter.setPen(pen)

        geom = self.geometry
        x, y = geom.x(), geom.y()
        w, h = geom.width(), geom.height()
        wsingle = w / len(self.dist)
        for d, c in zip(self.dist, self.colors):
            painter.setBrush(QBrush(c))
            painter.drawRect(QRectF(x, y, wsingle, d * h))
            x += wsingle
        painter.end()

        self.__picture = picture


class DistributionTest(OWWidget):
    name = 'Distribution Test'
    description = 'Check if data is in given distribution.'
    icon = 'icons/mywidget.svg'
    want_main_area = True
    buttons_area_orientation = Qt.Vertical
    resizing_enabled = True
    inputs = [('Data', Orange.data.Table, 'set_data')]

    available_tests = (
        KolmogorovSmirnov,
        AndersonDarling,
        ShapiroWilk,
        ChiSquare,
    )
    settingsHandler = settings.DomainContextHandler(
        match_values=settings.DomainContextHandler.MATCH_VALUES_ALL)
    #: Selected variable index
    available_distributions = [d for d in Distribution]
    test_idx = 0
    distribution_idx = 0
    column_idx = 0
    own_distribution_idx = 0

    relative_freq = settings.Setting(False)

    smoothing_index = settings.Setting(5)
    show_prob = settings.ContextSetting(0)

    graph_name = "box_scene"

    ASH_HIST = 50

    bins = [2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 50]

    def __init__(self):
        super().__init__()
        self.varmodel = itemmodels.VariableListModel()
        self.groupvarmodel = []
        self.distributions = [distribution.value
                              for distribution
                              in self.available_distributions]
        box = gui.vBox(self.controlArea, 'Tests')
        gui.radioButtonsInBox(
            box, self, 'test_idx',
            btnLabels=[test.name for test in self.available_tests],
            callback=self.test_changed,
        )

        box = gui.vBox(self.controlArea, 'Distributions')
        self.distribution_choose = gui.radioButtonsInBox(
            box, self, 'distribution_idx',
            btnLabels=self.distributions,
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
        self.infolabel = gui.widgetLabel(box, "<center>p-value: </center>")
        self.mainArea.setMinimumWidth(800)
        self.own_distribution_choose = gui.comboBox(
            self.controlArea, self, 'own_distribution_idx',
            box='Own distribution',
            items=[],
            orientation=Qt.Horizontal,
            callback=self.column_changed,
        )

        self.own_distribution_choose.setModel(self.available_columns)
        self.data = None

        self.plotview = pg.PlotWidget(background=None)
        self.plotview.setRenderHint(QPainter.Antialiasing)
        self.mainArea.layout().addWidget(self.plotview)
        w = QLabel()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mainArea.layout().addWidget(w, Qt.AlignCenter)
        self.ploti = pg.PlotItem()
        self.box_scene = self.ploti.vb
        self.ploti.hideButtons()
        self.plotview.setCentralItem(self.ploti)

        self.plot_prob = pg.ViewBox()
        self.ploti.scene().addItem(self.plot_prob)
        self.ploti.getAxis("right").linkToView(self.plot_prob)
        self.ploti.getAxis("right").setLabel("Probability")
        self.plot_prob.setZValue(10)
        self.plot_prob.setXLink(self.ploti)
        self.update_views()
        self.ploti.vb.sigResized.connect(self.update_views)
        self.plot_prob.setRange(yRange=[0, 1])

        def disable_mouse(box_scene):
            box_scene.setMouseEnabled(False, False)
            box_scene.setMenuEnabled(False)

        disable_mouse(self.box_scene)
        disable_mouse(self.plot_prob)

        self.tooltip_items = []

        pen = QPen(self.palette().color(QPalette.Text))
        for axis in ("left", "bottom"):
            self.ploti.getAxis(axis).setPen(pen)

        self._legend = LegendItem()
        self._legend.setParentItem(self.box_scene)
        self._legend.hide()
        self._legend.anchor((1, 0), (1, 0))
        self.test_changed()

    def update_views(self):
        """
        resize
        """
        self.plot_prob.setGeometry(self.box_scene.sceneBoundingRect())
        self.plot_prob.linkedViewChanged(self.box_scene, self.plot_prob.XAxis)

    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.warning()
        self.data = data
        self.own_distribution_choose.hide()
        if data is not None:
            self.available_columns[:] = data.domain
            domain = self.data.domain

            self.varmodel[:] = list(domain) + [
                meta for meta in domain.metas
                if meta.is_continuous or meta.is_discrete
                ]
            self.groupvarmodel = \
                ["(None)"] + [var for var in domain if var.is_discrete] + \
                [meta for meta in domain.metas if meta.is_discrete]
            if domain.has_discrete_class:
                self.groupvar_idx = \
                    self.groupvarmodel[1:].index(domain.class_var) + 1
            self.openContext(domain)
            self.column_idx = min(max(self.column_idx, 0),
                                  len(self.varmodel) - 1)
            self.groupvar_idx = min(max(self.groupvar_idx, 0),
                                    len(self.groupvarmodel) - 1)
            self._setup()

    def test_changed(self):
        """
        Management of buttons, depends from type of distribution test.
        Own samples are hidden.
        :return:
        """
        for idx, button in enumerate(self.distribution_choose.buttons):
            if Distribution(button.text()) in self.test.allowed_distribution:
                button.show()

                if self.distribution not in self.test.allowed_distribution:
                    button.toggle()
                    self.distribution_idx = idx
            else:
                self.own_distribution_choose.hide()
                button.hide()
        self.compute_p_value()

    def distribution_changed(self):
        """
         Control buttons of allowed distributions - show or hide
        :return: compute p-value
        """
        if self.distribution == Distribution.OWN:
            self.own_distribution_choose.show()
        else:
            self.own_distribution_choose.hide()
        self.compute_p_value()

    def clear(self):
        self.box_scene.clear()
        self.plot_prob.clear()
        self.varmodel[:] = []
        self.groupvarmodel = []
        self.column_idx = -1
        self.groupvar_idx = 0
        self._legend.clear()
        self._legend.hide()

    def column_changed(self):
        """
        compute p-value if column is changed
        """
        self._setup()
        self.compute_p_value()

    def _setup(self):
        """
        set new plot
        """
        self.box_scene.clear()
        self.plot_prob.clear()
        self._legend.clear()
        self._legend.hide()

        varidx = self.column_idx
        self.var = self.cvar = None
        if varidx >= 0:
            self.var = self.varmodel[varidx]

        data = self.data
        if self.var is None:
            return
        self.set_left_axis_name()
        if self.cvar:

            self.contingencies = \
                contingency.get_contingency(data, self.var, self.cvar)
            self.display_contingency()
        else:

            self.distributions = \
                distribution.get_distribution(data, self.var)
            self.display_distribution()
        self.box_scene.autoRange()

    def compute_p_value(self):
        """
        :return: p-value
        """
        if self.data is not None:
            p_value = self.test.compute(self)
            if isinstance(p_value, float):
                self.infolabel.setText(
                    "\n".join(["p-value:" + str(round(p_value, 3))]))
            else:

                self.infolabel.setText(
                    '\np-value: {}'.format(round(p_value, 3)))

    def _on_relative_freq_changed(self):
        self.set_left_axis_name()
        if self.cvar and self.cvar.is_discrete:
            self.display_contingency()
        else:
            self.display_distribution()
        self.box_scene.autoRange()

    def display_contingency(self):
        """
        Set the contingency to display.
        """
        cont = self.contingencies
        var, cvar = self.var, self.cvar
        assert len(cont) > 0
        self.box_scene.clear()
        self.plot_prob.clear()
        self._legend.clear()
        self.tooltip_items = []

        if self.show_prob:
            self.ploti.showAxis('right')
        else:
            self.ploti.hideAxis('right')

        bottomaxis = self.ploti.getAxis("bottom")
        bottomaxis.setLabel(var.name)
        bottomaxis.resizeEvent()

        cvar_values = cvar.values
        colors = [QColor(*col) for col in cvar.colors]

        if var and var.is_continuous:

            bottomaxis.setTicks(None)

            weights, cols, cvar_values, curves = [], [], [], []
            for i, dist in enumerate(cont):
                v, W = dist
                if len(v):
                    weights.append(np.sum(W))
                    cols.append(colors[i])
                    cvar_values.append(cvar.values[i])
                    curves.append(
                        ash_curve(dist, cont, m=DistributionTest.ASH_HIST))
            weights = np.array(weights)
            sumw = np.sum(weights)
            weights /= sumw
            colors = cols
            curves = [(X, Y * w) for (X, Y), w in zip(curves, weights)]

            curvesline = []  # from histograms to lines
            for (X, Y) in curves:
                X += np.array(((X[1] - X[0]) / 2)[:-1])
                Y = np.array(Y)
                curvesline.append((X, Y))

            for t in ["fill", "line"]:
                for (X, Y), color, w, cval in reversed(
                        list(zip(curvesline, colors, weights, cvar_values))):
                    item = pg.PlotCurveItem()
                    pen = QPen(QBrush(color), 3)
                    pen.setCosmetic(True)
                    color = QColor(color)
                    color.setAlphaF(0.2)
                    item.setData(X, Y / (w if self.relative_freq else 1),
                                 antialias=True, stepMode=False,
                                 fillLevel=0 if t == "fill" else None,
                                 brush=QBrush(color), pen=pen)
                    self.box_scene.addItem(item)
                    if t == "line":
                        if self.relative_freq:
                            density = "Normalized density"
                        else:
                            density = "Density"
                        item.tooltip = "{density}\n{name}={value}".format(
                            value=cval, name=cvar.name, density=density)
                        self.tooltip_items.append((self.box_scene, item))

            if self.show_prob:
                all_X = np.array(
                    np.unique(np.hstack([X for X, _ in curvesline])))
                inter_X = np.array(
                    np.linspace(all_X[0], all_X[-1], len(all_X) * 2))
                curvesinterp = [np.interp(inter_X, X, Y) for (X, Y) in
                                curvesline]
                sumprob = np.sum(curvesinterp, axis=0)
                legal = sumprob > 0.05 * np.max(sumprob)

                i = len(curvesinterp) + 1
                show_all = self.show_prob == i
                for Y, color, cval in reversed(
                        list(zip(curvesinterp, colors, cvar_values))):
                    i -= 1
                    if show_all or self.show_prob == i:
                        item = pg.PlotCurveItem()
                        pen = QPen(QBrush(color), 3, style=Qt.DotLine)
                        pen.setCosmetic(True)
                        prob = Y[legal] / sumprob[legal]
                        item.setData(inter_X[legal], prob, antialias=True,
                                     stepMode=False,
                                     fillLevel=None, brush=None, pen=pen)
                        self.plot_prob.addItem(item)
                        item.tooltip = \
                            "Probability that \n{name}={value}".format(
                                name=cvar.name, value=cval)
                        self.tooltip_items.append((self.plot_prob, item))

        elif var and var.is_discrete:

            bottomaxis.setTicks([list(enumerate(var.values))])

            cont = np.array(cont)

            maxh = 0  # maximal column height
            maxrh = 0  # maximal relative column height
            scvar = cont.sum(axis=1)
            # a cvar with sum=0 with allways have distribution counts 0,
            # therefore we can divide it by anything
            scvar[scvar == 0] = 1
            for i, (value, dist) in enumerate(zip(var.values, cont.T)):
                maxh = max(maxh, max(dist))
                maxrh = max(maxrh, max(dist / scvar))

            for i, (value, dist) in enumerate(zip(var.values, cont.T)):
                dsum = sum(dist)
                geom = QRectF(
                    i - 0.333, 0, 0.666, maxrh
                    if self.relative_freq else maxh)
                item = DistributionBarItem(
                    geom, dist / scvar / maxrh
                    if self.relative_freq
                    else dist / maxh, colors)
                self.box_scene.addItem(item)
                tooltip = "\n".join(
                    "%s: %.*f" % (n, 3 if self.relative_freq else 1, v)
                    for n, v in zip(
                        cvar_values,
                        dist / scvar if self.relative_freq else dist))
                if self.relative_freq:
                    frequency = "Normalized frequency"
                else:
                    frequency = "Frequency"
                item.tooltip = \
                    "{frequency}({name}={value}):\n{tooltip}".format(
                        frequency=frequency,
                        name=cvar.name,
                        value=value,
                        tooltip=tooltip,
                    )
                self.tooltip_items.append((self.box_scene, item))

                if self.show_prob:
                    item.tooltip += "\n\nProbabilities:"
                    for ic, a in enumerate(dist):
                        if (self.show_prob - 1 != ic
                            and self.show_prob - 1 != len(dist)) \
                                or dsum < 1e-6:
                            continue
                        position = -0.333 + ((ic + 0.5) * 0.666 / len(dist))
                        prob = a / dsum
                        if not 1e-6 < prob < 1 - 1e-6:
                            continue
                        ci = 1.96 * sqrt(prob * (1 - prob) / dsum)
                        item.tooltip += "\n%s: %.3f ± %.3f" % (
                            cvar_values[ic], prob, ci)
                        mark = pg.ScatterPlotItem()
                        bar = pg.ErrorBarItem()
                        pen = QPen(QBrush(QColor(0)), 1)
                        pen.setCosmetic(True)
                        bar.setData(x=[i + position], y=[prob],
                                    bottom=min(np.array([ci]), prob),
                                    top=min(np.array([ci]), 1 - prob),
                                    beam=np.array([0.05]),
                                    brush=QColor(1), pen=pen)
                        mark.setData([i + position], [prob], antialias=True,
                                     symbol="o",
                                     fillLevel=None, pxMode=True, size=10,
                                     brush=QColor(colors[ic]), pen=pen)
                        self.plot_prob.addItem(bar)
                        self.plot_prob.addItem(mark)

        for color, name in zip(colors, cvar_values):
            self._legend.addItem(
                ScatterPlotItem(pen=color, brush=color, size=10, shape="s"),
                escape(name)
            )
        self._legend.show()

    def set_left_axis_name(self):
        leftaxis = self.ploti.getAxis("left")
        set_label = leftaxis.setLabel
        if self.var and self.var.is_continuous:

            set_label(["Density", "Relative density"]
                      [self.cvar is not None and self.relative_freq])
        else:

            set_label(["Frequency", "Relative frequency"]
                      [self.cvar is not None and self.relative_freq])
        leftaxis.resizeEvent()

    def display_distribution(self):
        dist = self.distributions
        var = self.var
        assert len(dist) > 0
        self.box_scene.clear()
        self.plot_prob.clear()
        self.ploti.hideAxis('right')
        self.tooltip_items = []

        bottomaxis = self.ploti.getAxis("bottom")
        bottomaxis.setLabel(var.name)
        bottomaxis.resizeEvent()

        self.set_left_axis_name()
        if var and var.is_continuous:

            bottomaxis.setTicks(None)
            if not len(dist[0]):
                return
            edges, curve = ash_curve(dist, None, m=DistributionTest.ASH_HIST)
            edges = edges + (edges[1] - edges[0]) / 2
            edges = edges[:-1]
            item = pg.PlotCurveItem()
            pen = QPen(QBrush(Qt.black), 3)
            pen.setCosmetic(True)
            item.setData(edges, curve, antialias=True, stepMode=False,
                         fillLevel=0, brush=QBrush(Qt.gray), pen=pen)
            self.box_scene.addItem(item)
            item.tooltip = "Density"
            self.tooltip_items.append((self.box_scene, item))
        else:

            bottomaxis.setTicks([list(enumerate(var.values))])
            for i, w in enumerate(dist):
                geom = QRectF(i - 0.33, 0, 0.66, w)
                item = DistributionBarItem(geom, [1.0],
                                           [QColor(128, 128, 128)])
                self.box_scene.addItem(item)
                item.tooltip = "Frequency for %s: %r" % (var.values[i], w)
                self.tooltip_items.append((self.box_scene, item))

    def onDeleteWidget(self):
        self.box_scene.clear()
        super().onDeleteWidget()

    def get_widget_name_extension(self):
        if self.column_idx >= 0:
            return self.varmodel[self.column_idx]

    @property
    def test(self) -> Test:
        return self.available_tests[self.test_idx]

    @property
    def distribution(self) -> Distribution:
        return self.available_distributions[self.distribution_idx]

    @property
    def column(self):
        return self.data[:, self.column_idx]

    @property
    def own_distribution(self):
        return self.data[:, self.own_distribution_idx]


def selected_index(view):
    """Return the selected integer `index` (row) in the view.

    If no index is selected return -1

    `view` must be in single selection mode.
    """
    indices = view.selectedIndexes()
    assert len(indices) < 2, "View must be in single selection mode"
    if indices:
        return indices[0].row()
    else:
        return -1


def dist_sum(D1, D2):
    """
    A sum of two continuous distributions.
    """
    X1, W1 = D1
    X2, W2 = D2
    X = np.r_[X1, X2]
    W = np.r_[W1, W2]
    sort_ind = np.argsort(X)
    X, W = X[sort_ind], W[sort_ind]

    unique, uniq_index = np.unique(X, return_index=True)
    spans = np.diff(np.r_[uniq_index, len(X)])
    W = [np.sum(W[start:start + span])
         for start, span in zip(uniq_index, spans)]
    W = np.array(W)
    assert W.shape[0] == unique.shape[0]
    return unique, W


def ash_curve(dist, cont=None, m=3):
    """
    To histogram
    :m: Number of shifted histograms.
    :return: edge and histogram
    """
    dist = np.asarray(dist)
    X, W = dist
    std = weighted_std(X, weights=W)
    size = X.size
    # if only one sample in the class
    if std == 0 and cont is not None:
        std = weighted_std(cont.values,
                           weights=np.sum(cont.counts, axis=0))
        size = cont.values.size
    # if attr is constant or contingencies is None (no class variable)
    if std == 0:
        std = 0.1
        size = X.size
    bandwidth = 3.5 * std * (size ** (-1 / 3))

    hist, edges = average_shifted_histogram(X, bandwidth, m, weights=W)
    return edges, hist


def average_shifted_histogram(a, h, m=3, weights=None):
    """
    Compute the average shifted histogram.

    Parameters
    ----------
    a : array-like
        Input data.
    h : float
        Base bin width.
    m : int
        Number of shifted histograms.
    weights : array-like
        An array of weights of the same shape as `a`
    """
    a = np.asarray(a)
    smoothing = 1
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError("weights should have the same shape as a")
        weights = weights.ravel()

    a = a.ravel()

    amin, amax = a.min(), a.max()
    h = h * 0.5 * smoothing
    delta = h / m
    wfac = 4  # extended windows for gaussian smoothing
    offset = (wfac * m - 1) * delta
    nbins = max(np.ceil((amax - amin + 2 * offset) / delta),
                2 * m * wfac - 1)

    bins = np.linspace(amin - offset, amax + offset, nbins + 1,
                       endpoint=True)
    hist, edges = np.histogram(a, bins, weights=weights, density=True)

    kernel = gaussian_kernel(
        (np.arange(2 * wfac * m - 1) - (wfac * m - 1)) / (wfac * m), wfac)
    kernel = kernel / np.sum(kernel)
    ash = np.convolve(hist, kernel, mode="same")

    ash = ash / np.diff(edges) / ash.sum()
    return ash, edges


def gaussian_kernel(x, k):
    # fit k standard deviations into available space from [-1 .. 1]
    return 1 / (np.sqrt(2 * np.pi)) * np.exp(- (x * k) ** 2 / (2))


def weighted_std(a, axis=None, weights=None, ddof=0):
    mean = np.average(a, axis=axis, weights=weights)

    if axis is not None:
        shape = shape_reduce_keep_dims(a.shape, axis)
        mean = mean.reshape(shape)

    sq_diff = np.power(a - mean, 2)
    mean_sq_diff, wsum = np.average(
        sq_diff, axis=axis, weights=weights, returned=True
    )

    if ddof != 0:
        mean_sq_diff *= wsum / (wsum - ddof)

    return np.sqrt(mean_sq_diff)


def shape_reduce_keep_dims(shape, axis):
    if shape is None:
        return ()

    shape = list(shape)
    if isinstance(axis, collections.Sequence):
        for ax in axis:
            shape[ax] = 1
    else:
        shape[axis] = 1
    return tuple(shape)


def main(argv=None):
    import gc
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    w = DistributionTest()
    w.show()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "heart_disease"
    data = Orange.data.Table(filename)
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval


if __name__ == "__main__":
    sys.exit(main())
