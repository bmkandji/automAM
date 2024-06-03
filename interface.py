import sys
import json
import random
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QComboBox, QListWidget, QCheckBox, QListWidgetItem, QGroupBox, QSlider, \
    QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from main import main  # Assurez-vous que votre fonction main ne bloque pas ou utilise QTimer


class EmittingStream(QObject):
    text_written = pyqtSignal(str)  # Définir un signal personnalisé

    def write(self, text):
        self.text_written.emit(str(text))  # Émettre le texte comme signal

    def flush(self):
        pass  # Cela peut être nécessaire pour la compatibilité avec l'interface de type fichier


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)
        plt.tight_layout()

    def plot(self, data):
        self.ax.clear()
        self.ax.plot(data['x'], data['y'], label='Portfolio Trajectory')
        self.ax.plot(data['x'], data['value'], label='Portfolio Value')
        self.ax.plot(data['x'], data['pl'], label='P&L')
        self.ax.legend()
        self.draw()


class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setStyleSheet(self.get_styles())
        self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_plot)

    def initUI(self):
        self.setWindowTitle('Run Portfolio Manager')
        layout = QVBoxLayout()

        self.comboBoxList = QListWidget()
        options = ["Default Settings", "Option 1", "Option 2", "Option 3"]
        for option in options:
            item = QListWidgetItem(option)
            checkbox = QCheckBox()
            checkbox.setText(option)
            checkbox.stateChanged.connect(self.update_option_visibility)
            self.comboBoxList.addItem(item)
            self.comboBoxList.setItemWidget(item, checkbox)

        self.comboBox = QComboBox()
        self.comboBox.setModel(self.comboBoxList.model())
        self.comboBox.setView(self.comboBoxList)

        self.btn_start = QtWidgets.QPushButton('Start', self)
        self.btn_start.clicked.connect(self.start_script)

        self.log_widget = QtWidgets.QTextEdit(self)
        self.log_widget.setReadOnly(True)

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.setValue(50)
        self.slider1.setTickPosition(QSlider.TicksBelow)
        self.slider1.setTickInterval(10)

        self.textbox1 = QtWidgets.QLineEdit(self)

        self.option1Group = QGroupBox("Option 1 Settings")
        self.option2Group = QGroupBox("Option 2 Settings")
        self.option3Group = QGroupBox("Option 3 Settings")

        self.option1Slider = QSlider(Qt.Horizontal)
        self.option1Slider.setMinimum(0)
        self.option1Slider.setMaximum(100)
        self.option1Slider.setTickPosition(QSlider.TicksBelow)
        self.option1Slider.setTickInterval(10)

        self.option2Slider = QSlider(Qt.Horizontal)
        self.option2Slider.setMinimum(0)
        self.option2Slider.setMaximum(50)
        self.option2Slider.setTickPosition(QSlider.TicksBelow)
        self.option2Slider.setTickInterval(5)

        self.option3Checkbox1 = QCheckBox("Sub Option 1")
        self.option3Checkbox2 = QCheckBox("Sub Option 2")
        self.option3Checkbox3 = QCheckBox("Sub Option 3")

        self.canvas = PlotCanvas(self, width=8, height=6)

        option1Layout = QVBoxLayout()
        option1Layout.addWidget(QLabel("Aversion:"))
        option1Layout.addWidget(self.option1Slider)
        self.option1Group.setLayout(option1Layout)

        option2Layout = QVBoxLayout()
        option2Layout.addWidget(QLabel("Another Slider:"))
        option2Layout.addWidget(self.option2Slider)
        self.option2Group.setLayout(option2Layout)

        option3Layout = QVBoxLayout()
        option3Layout.addWidget(self.option3Checkbox1)
        option3Layout.addWidget(self.option3Checkbox2)
        option3Layout.addWidget(self.option3Checkbox3)
        self.option3Group.setLayout(option3Layout)

        layout.addWidget(self.comboBox)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.log_widget)
        layout.addWidget(self.slider1)
        layout.addWidget(self.textbox1)
        layout.addWidget(self.option1Group)
        layout.addWidget(self.option2Group)
        layout.addWidget(self.option3Group)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.update_option_visibility()

    def get_styles(self):
        return """
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 5px;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #fff;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #4CAF50;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """

    def update_option_visibility(self):
        self.option1Group.setVisible(False)
        self.option2Group.setVisible(False)
        self.option3Group.setVisible(False)

        for i in range(self.comboBoxList.count()):
            item = self.comboBoxList.item(i)
            checkbox = self.comboBoxList.itemWidget(item)
            if checkbox.isChecked():
                if checkbox.text() == "Option 1":
                    self.option1Group.setVisible(True)
                elif checkbox.text() == "Option 2":
                    self.option2Group.setVisible(True)
                elif checkbox.text() == "Option 3":
                    self.option3Group.setVisible(True)

    def start_script(self):
        data = {
            "slider_value": self.slider1.value(),
            "textbox_value": self.textbox1.text(),
            "option1_input": self.option1Slider.value() if self.option1Group.isVisible() else "",
            "option2_input": self.option2Slider.value() if self.option2Group.isVisible() else "",
            "option3_input": {
                "sub_option1": self.option3Checkbox1.isChecked(),
                "sub_option2": self.option3Checkbox2.isChecked(),
                "sub_option3": self.option3Checkbox3.isChecked(),
            } if self.option3Group.isVisible() else {}
        }

        with open("config.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

        sys.stdout = EmittingStream(text_written=self.on_text_written)

        self.run_main_with_timer()  # Utiliser QTimer pour appeler la fonction main sans bloquer

        self.timer.start(1000)  # Démarrer le timer pour mettre à jour le graphique toutes les secondes

    def run_main_with_timer(self):
        QTimer.singleShot(0, self.run_main)

    def run_main(self):
        try:
            main()  # Appeler directement la fonction main
        except Exception as e:
            self.on_thread_error(str(e))
        finally:
            self.on_thread_finished()

    def on_text_written(self, text):
        self.log_widget.moveCursor(QtGui.QTextCursor.End)
        self.log_widget.insertPlainText(text)
        QtWidgets.QApplication.processEvents()

    def on_thread_finished(self):
        self.plot_results()
        self.timer.stop()  # Arrêter le timer lorsque la fonction est terminée

    def on_thread_error(self, error_message):
        self.log_widget.moveCursor(QtGui.QTextCursor.End)
        self.log_widget.insertPlainText(f"Error: {error_message}\n")
        QtWidgets.QApplication.processEvents()

    def plot_example_data(self):
        data = {
            "x": range(10),
            "y": [random.randint(0, 100) for _ in range(10)],
            "value": [random.randint(0, 100) for _ in range(10)],
            "pl": [random.randint(0, 100) for _ in range(10)],
        }
        self.canvas.plot(data)

    def update_plot(self):
        self.plot_example_data()  # Mettre à jour le graphique avec de nouvelles données périodiquement

    def plot_results(self):
        self.plot_example_data()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
