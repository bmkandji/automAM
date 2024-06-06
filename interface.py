import sys
import json
import random
import threading
import time
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QComboBox, QListWidget, QCheckBox, QListWidgetItem, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import rpy2.robjects as robjects
import contextvars
from main import main  # Assuming main is a function you want to run

# Assume this is the context variable used by rpy2
rpy2_context_var = contextvars.ContextVar('rpy2_context_var')

# Set up the rpy2 conversion rules
def initialize_rpy2():
    robjects.conversion.get_conversion()

# Initialize rpy2 conversion rules in the main context
rpy2_context_var.set(initialize_rpy2())

class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass

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
        self.stop_flag = False  # Flag to indicate if the script should be stopped
        self.script_thread = None  # Thread to run the main script

        # Initialize the rpy2 context for the main thread
        initialize_rpy2()

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
        self.comboBox0 = QComboBox()
        self.comboBox0.setModel(self.comboBoxList.model())
        self.comboBox0.setView(self.comboBoxList)

        self.comboBox = QComboBox()
        self.comboBox.addItems(["Default Settings", "Mean-Var", "Traking-Error", "Targeting-vol"])
        self.comboBox.currentIndexChanged.connect(self.update_option_visibility)

        # Boutons Start et Stop
        self.btn_start = QtWidgets.QPushButton('Start', self)
        self.btn_start.clicked.connect(self.start_script)
        self.btn_stop = QtWidgets.QPushButton('Stop', self)
        self.btn_stop.clicked.connect(self.stop_script)
        self.btn_stop.setObjectName("btn_stop")  # Set object name for styling

        # Disposition horizontale pour les boutons
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_stop)

        # Slider gradué de 1 à 100
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.setValue(50)
        self.slider1.setTickPosition(QSlider.TicksBelow)
        self.slider1.setTickInterval(1)

        # Étiquette pour afficher la valeur actuelle du slider
        self.slider1_value_label = QLabel(f"Slider Value: {self.slider1.value()}")
        self.slider1.valueChanged.connect(self.update_slider_value_label)

        self.option1Group = QGroupBox("Mean-Var Settings")
        self.option2Group = QGroupBox("Traking-Error Settings")
        self.option3Group = QGroupBox("Targeting-vol Settings")

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

        self.log_widget = QtWidgets.QTextEdit(self)
        self.log_widget.setReadOnly(True)

        layout.addWidget(self.comboBox0)
        layout.addWidget(self.comboBox)
        layout.addLayout(buttons_layout)  # Ajouter la disposition des boutons
        layout.addWidget(self.log_widget)
        layout.addWidget(self.slider1)
        layout.addWidget(self.slider1_value_label)  # Ajouter l'étiquette pour la valeur du slider
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
            QPushButton#btn_stop {
                background-color: #FFEB3B;
                color: black;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton#btn_stop:hover {
                background-color: #FDD835;
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
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border: 1px solid #4CAF50;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #fff;
                border: 1px solid #bbb;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::tick-mark:horizontal {
                background: black;
                height: 5px;
                width: 1px;
            }
        """

    def update_option_visibility(self):
        self.option1Group.setVisible(False)
        self.option2Group.setVisible(False)
        self.option3Group.setVisible(False)

        selected_option = self.comboBox.currentText()
        if selected_option == "Mean-Var":
            self.option1Group.setVisible(True)
        elif selected_option == "Traking-Error":
            self.option2Group.setVisible(True)
        elif selected_option == "Targeting-vol":
            self.option3Group.setVisible(True)

    def update_slider_value_label(self):
        self.slider1_value_label.setText(f"Slider Value: {self.slider1.value()}")

    def start_script(self):
        self.stop_flag = False  # Reset the stop flag when starting the script
        data = {
            "slider_value": self.slider1.value(),
            "option1_input": self.option1Slider.value() if self.option1Group.isVisible() else "",
            "option2_input": self.option2Slider.value() if self.option2Group.isVisible() else "",
            "option3_input": [
                self.option3Checkbox1.isChecked(),
                self.option3Checkbox2.isChecked(),
                self.option3Checkbox3.isChecked()
            ] if self.option3Group.isVisible() else ""
        }
        with open("data.json", "w") as file:
            json.dump(data, file)

        sys.stdout = EmittingStream(text_written=self.on_text_written)
        sys.stderr = EmittingStream(text_written=self.on_text_written)

        # Create a new context for the thread
        thread_context = contextvars.copy_context()
        self.script_thread = threading.Thread(target=self.run_main, args=(thread_context,))
        self.script_thread.start()

        self.timer.start(1000)  # Mettre à jour le graphique toutes les secondes

    def stop_script(self):
        self.stop_flag = True  # Set the stop flag to True to indicate the script should stop
        if self.script_thread is not None:
            self.script_thread.join()  # Wait for the thread to finish
        self.timer.stop()
        self.log_widget.clear()

    def run_main(self, ctx):
        # Run the main function within the provided context
        ctx.run(self._run_main)

    def _run_main(self):
        while not self.stop_flag:
            try:
                main()  # Appeler directement la fonction main
                time.sleep(1)  # Ajouter un délai pour simuler un travail en cours
            except Exception as e:
                self.on_thread_error(str(e))
                break
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
