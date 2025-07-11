"""# PyQt6 local example

Show how to integrate offscreen Datoviz figures into a PyQt6 application, using the Datoviz
server API which provides a fully offscreen renderer with support for multiple canvases.

NOTE: this API is experimental and will change in an upcoming release.

"""

import sys

import numpy as np

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter
except:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter

import datoviz as dvz
from datoviz.backends.pyqt6 import QtServer

WIDTH, HEIGHT = 800, 600


class ExampleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Example Qt Datoviz window")

        # Create a Qt Datoviz server.
        self.qt_server = QtServer()

        # Create two figures (special Qt widgets with a Datoviz figure).
        w, h = WIDTH // 2, HEIGHT
        self.qfig1 = self.qt_server.figure(w, h)
        self.qfig2 = self.qt_server.figure(w, h)

        panel1 = self.qfig1.panel((0, 0), (w, h))
        panel1.demo_3D()

        panel2 = self.qfig2.panel((0, 0), (w, h))
        N = 5
        colors = dvz.cmap('spring', np.linspace(0, 1, N))
        scale = 0.35
        sc = dvz.ShapeCollection()
        sc.add_tetrahedron(offset=(-1, 0.5, -0.5), scale=scale, color=colors[0])
        sc.add_hexahedron(offset=(0, 0.5, -0.5), scale=scale, color=colors[1])
        sc.add_octahedron(offset=(1, 0.5, -0.5), scale=scale, color=colors[2])
        sc.add_dodecahedron(offset=(-0.5, -0.5, 0), scale=scale, color=colors[3])
        sc.add_icosahedron(offset=(+0.5, -0.5, 0), scale=scale, color=colors[4])
        panel2.arcball()
        visual = self.qt_server.mesh(sc, lighting=True)
        panel2.add(visual)

        # Add the two figures in the main window.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.qfig1)
        splitter.addWidget(self.qfig2)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        self.setCentralWidget(splitter)
        self.resize(WIDTH, HEIGHT)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = ExampleWindow()
    mw.show()
    sys.exit(app.exec())
