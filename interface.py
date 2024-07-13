import sys
import json
import random
import threading
import time
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QComboBox, QListWidget, QCheckBox, QListWidgetItem, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout, QTextEdit
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import rpy2.robjects as robjects
import contextvars
from utils.load import load_json_config
from utils.input import interface_input
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
        #self.timer = QTimer(self)
        self.stop_event = threading.Event()  # Event to signal the thread to stop
        self.script_thread = None  # Thread to run the main script

        # Initialize the rpy2 context for the main thread
        initialize_rpy2()

    def initUI(self):
        self.setWindowTitle('Run Portfolio Manager')
        layout = QVBoxLayout()

        # Group for Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        settings_group.setLayout(settings_layout)

        self.comboBoxList = QListWidget()
        self.comboBoxList.setSelectionMode(QListWidget.SingleSelection)
        options = ["Default Settings", "AAPL", "AMZN", "GOOGL", "MSFT"]
        for option in options:
            item = QListWidgetItem(option)
            checkbox = QCheckBox()
            checkbox.setText(option)
            checkbox.stateChanged.connect(self.handle_asset_selection)
            self.comboBoxList.addItem(item)
            self.comboBoxList.setItemWidget(item, checkbox)
        self.comboBox0 = QComboBox()
        self.comboBox0.setModel(self.comboBoxList.model())
        self.comboBox0.setView(self.comboBoxList)

        self.comboBox = QComboBox()
        self.comboBox.addItems(["Default Settings", "Mean-Var",
                                "Targeting-vol", "Tracking-Error"])
        self.comboBox.currentIndexChanged.connect(self.update_option_visibility)

        settings_layout.addRow("Select Assets:", self.comboBox0)
        settings_layout.addRow("Select Strategy:", self.comboBox)

        # Sliders and checkboxes for options
        self.option1Slider = QSlider(Qt.Horizontal)
        self.option1Slider.setMinimum(0)
        self.option1Slider.setMaximum(100)
        self.option1Slider.setTickPosition(QSlider.TicksBelow)
        self.option1Slider.setTickInterval(1)
        self.option1Slider_label = QLabel(f"Current Value: {self.option1Slider.value()}")
        self.option1Slider.valueChanged.connect(self.update_option1_slider_value_label)

        self.option2Slider = QSlider(Qt.Horizontal)
        self.option2Slider.setMinimum(0)
        self.option2Slider.setMaximum(100)
        self.option2Slider.setTickPosition(QSlider.TicksBelow)
        self.option2Slider.setTickInterval(1)
        self.option2Slider_label = QLabel(f"Current Value: {self.option2Slider.value()}")
        self.option2Slider.valueChanged.connect(self.update_option2_slider_value_label)

        self.option3ComboBox = QComboBox()
        self.option3ComboBox.addItems(["Eq_weighted", "Market_Index"])
        self.option3ComboBox.currentIndexChanged.connect(self.update_option_visibility)

        self.option1Group = QGroupBox("Aversion")
        self.option2Group = QGroupBox()
        self.option3Group = QGroupBox("Reference portfolios")

        option1Layout = QVBoxLayout()
        option1Layout.addWidget(self.option1Slider)
        option1Layout.addWidget(self.option1Slider_label)
        self.option1Group.setLayout(option1Layout)

        option2Layout = QVBoxLayout()
        option2Layout.addWidget(self.option2Slider)
        option2Layout.addWidget(self.option2Slider_label)
        option2Layout.addWidget(self.option3ComboBox)
        self.option2Group.setLayout(option2Layout)

        option3Layout = QVBoxLayout()
        option3Layout.addWidget(self.option3ComboBox)
        self.option3Group.setLayout(option3Layout)

        settings_layout.addRow(self.option1Group)
        settings_layout.addRow(self.option2Group)
        settings_layout.addRow(self.option3Group)

        # Bouton Start/Stop
        self.btn_start_stop = QtWidgets.QPushButton('Start', self)
        self.btn_start_stop.clicked.connect(self.toggle_script)
        self.btn_start_stop.setObjectName("btn_start_stop")  # Set object name for styling

        # Disposition horizontale pour les boutons
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.btn_start_stop)

        self.log_widget = QTextEdit(self)
        self.log_widget.setReadOnly(True)
        self.log_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.canvas = PlotCanvas(self, width=8, height=6)

        layout.addWidget(settings_group)
        layout.addLayout(buttons_layout)  # Ajouter la disposition des boutons
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.log_widget)
        layout.addWidget(QLabel("Graph:"))
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        # Sélectionner par défaut "Default Settings" et "Default Settings" pour la stratégie
        self.comboBoxList.itemWidget(self.comboBoxList.item(0)).setChecked(True)
        self.comboBox.setCurrentIndex(0)

        self.update_option_visibility()

    def get_styles(self):
        return """
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
            QPushButton#btn_start_stop {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton#btn_start_stop:hover {
                background-color: #45a049;
            }
            QPushButton#btn_start_stop[stop=true] {
                background-color: #FF0000;
                color: white;
            }
            QPushButton#btn_start_stop[stop=true]:hover {
                background-color: #FF4500;
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
        elif selected_option == "Tracking-Error":
            self.option2Group.setTitle("Max Error (%)")
            self.option2Group.setVisible(True)
            self.option3Group.setVisible(True)
        elif selected_option == "Targeting-vol":
            self.option2Group.setTitle("Target Vol (%)")
            self.option2Group.setVisible(True)
            self.option3Group.setVisible(True)

    def handle_asset_selection(self):
        for i in range(self.comboBoxList.count()):
            item = self.comboBoxList.item(i)
            checkbox = self.comboBoxList.itemWidget(item)
            if checkbox.text() == "Default Settings" and checkbox.isChecked():
                for j in range(self.comboBoxList.count()):
                    other_item = self.comboBoxList.item(j)
                    other_checkbox = self.comboBoxList.itemWidget(other_item)
                    if other_checkbox.text() != "Default Settings":
                        other_checkbox.setChecked(False)
                        other_checkbox.setEnabled(False)
            elif checkbox.text() == "Default Settings" and not checkbox.isChecked():
                for j in range(self.comboBoxList.count()):
                    other_item = self.comboBoxList.item(j)
                    other_checkbox = self.comboBoxList.itemWidget(other_item)
                    if other_checkbox.text() != "Default Settings":
                        other_checkbox.setEnabled(True)

    def update_option1_slider_value_label(self):
        self.option1Slider_label.setText(f"Current Value: {self.option1Slider.value()}")

    def update_option2_slider_value_label(self):
        self.option2Slider_label.setText(f"Current Value: {self.option2Slider.value()}")

    def toggle_script(self):
        if self.btn_start_stop.property('stop'):
            self.stop_script()
        else:
            self.start_script()

    def start_script(self):
        self.set_widgets_enabled(False)
        self.stop_event.clear()  # Reset the stop event when starting the script

        # Vérifier que toutes les valeurs visibles sont définies
        selected_assets = [self.comboBoxList.itemWidget(self.comboBoxList.item(i)).text()
                           for i in range(self.comboBoxList.count())
                           if self.comboBoxList.itemWidget(self.comboBoxList.item(i)).isChecked()]

        if not selected_assets:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select at least one asset in Select Assets.")
            self.set_widgets_enabled(True)
            return

        if self.comboBox.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a strategy in Select Strategy.")
            self.set_widgets_enabled(True)
            return

        if self.option2Group.isVisible() and self.option2Slider.value() == 0:
            QtWidgets.QMessageBox.warning(self, "Warning",
                                          "Please set a value for Another Slider in Tracking-Error Settings.")
            self.set_widgets_enabled(True)
            return
        if self.option3Group.isVisible() and self.option3ComboBox.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a Sub Option in Targeting-vol Settings.")
            self.set_widgets_enabled(True)
            return

        # Collect all widget values
        input_data = {
            "selected_assets": selected_assets,
            "selected_strategy": self.comboBox.currentText(),
            "aversion": self.option1Slider.value() if self.option1Group.isVisible() else "",
            "tol": self.option2Slider.value() if self.option2Group.isVisible() else "",
            "ref_portfolio": self.option3ComboBox.currentText() if self.option3Group.isVisible() else ""
        }

        sys.stdout = EmittingStream(text_written=self.on_text_written)
        sys.stderr = EmittingStream(text_written=self.on_text_written)
        # Load the configuration for the Portfolio Manager
        pm_config = load_json_config(r"src/pfManger_settings/pfMananger_settings.json")
        pm_config = interface_input(pm_config, input_data)
        # Create a new context for the thread
        thread_context = contextvars.copy_context()
        self.script_thread = threading.Thread(target=self.run_main,
                                              args=(pm_config, thread_context))
        self.script_thread.start()

        #self.timer.start(1000)  # Mettre à jour le graphique toutes les secondes

        self.btn_start_stop.setText('Stop')
        self.btn_start_stop.setProperty('stop', True)
        self.btn_start_stop.setStyle(self.btn_start_stop.style())

    def stop_script(self):
        self.stop_event.set()  # Set the stop event to signal the thread to stop
        if self.script_thread is not None:
            self.script_thread.join()  # Wait for the thread to finish
            self.script_thread = None  # Reset the script_thread to None
        #self.timer.stop()
        self.set_widgets_enabled(True)

        self.btn_start_stop.setText('Start')
        self.btn_start_stop.setProperty('stop', False)
        self.btn_start_stop.setStyle(self.btn_start_stop.style())

    def set_widgets_enabled(self, enabled):
        self.comboBoxList.setEnabled(enabled)
        self.comboBox0.setEnabled(enabled)
        self.comboBox.setEnabled(enabled)
        self.option1Slider.setEnabled(enabled)
        self.option2Slider.setEnabled(enabled)
        self.option3ComboBox.setEnabled(enabled)
        self.option1Group.setEnabled(enabled)
        self.option2Group.setEnabled(enabled)
        self.option3Group.setEnabled(enabled)

    def run_main(self, pm_config, ctx):
        # Run the main function within the provided context
        ctx.run(self._run_main, pm_config)

    def _run_main(self, pm_config):
        while not self.stop_event.is_set():
            try:
                main(pm_config)  # Appeler directement la fonction main
                for _ in range(10):  # Add this to periodically check for the stop event
                    if self.stop_event.is_set():
                        return  # Exit the function if the stop event is set
                    time.sleep(0.1)  # Ajouter un délai pour simuler un travail en cours
            except Exception as e:
                self.on_thread_error(str(e))
                break
        self.on_thread_finished()

    def on_text_written(self, text):
        cursor = self.log_widget.textCursor()
        at_bottom = cursor.atEnd()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        if at_bottom:
            self.log_widget.ensureCursorVisible()
        self.log_widget.setTextCursor(cursor)

    def on_thread_finished(self):
        self.plot_results()
        #self.timer.stop()  # Arrêter le timer lorsque la fonction est terminée

    def on_thread_error(self, error_message):
        cursor = self.log_widget.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(f"Error: {error_message}\n")
        self.log_widget.setTextCursor(cursor)
        self.log_widget.ensureCursorVisible()
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
