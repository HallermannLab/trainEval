import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import sys
import numpy as np


def show_trace(time, values, title="Electrophysiological Trace Viewer"):
    """
    Displays a trace in a PyQtGraph window.

    Parameters:
        time (array-like): Time vector (x-axis).
        values (array-like): Signal values (y-axis).
        title (str): Window title.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title=title)
    win.resize(800, 600)
    win.show()

    plot = win.addPlot(title="Trace Viewer")
    plot.plot(time, values, pen='y')

    plot.setMouseEnabled(x=True, y=True)
    plot.showGrid(x=True, y=True)

    plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
    plot.getViewBox().setLimits(xMin=min(time), xMax=max(time))

    app.exec_()


def show_two_traces(time, trace1, trace2, title="Two-Trace Viewer"):
    """
    Displays two traces in vertically stacked plots with synchronized time axis.

    Parameters:
        time (array-like): Shared time vector.
        trace1 (array-like): First trace (top).
        trace2 (array-like): Second trace (bottom).
        title (str): Window title.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title=title)
    win.resize(1000, 600)
    win.show()

    # First plot (top)
    p1 = win.addPlot(row=0, col=0, title="Trace 1")
    p1.plot(time, trace1, pen='c')
    p1.setMouseEnabled(x=True, y=True)
    p1.showGrid(x=True, y=True)

    # Second plot (bottom)
    p2 = win.addPlot(row=1, col=0, title="Trace 2")
    p2.plot(time, trace2, pen='m')
    p2.setMouseEnabled(x=True, y=True)
    p2.showGrid(x=True, y=True)

    # Link x-axes for synchronized zoom/pan
    p2.setXLink(p1)

    # Optional: set initial visible range
    for p in (p1, p2):
        p.getViewBox().setLimits(xMin=min(time), xMax=max(time))
        p.getViewBox().setMouseMode(pg.ViewBox.RectMode)  # rectangular zoom

    app.exec_()