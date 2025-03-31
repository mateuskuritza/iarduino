import sys
import os
import shutil
import subprocess
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QMessageBox,
    QInputDialog,
)
from predict import get_latest_model
from shared import PROCESSED_DATA_DIR, CAPTURES_DIR


def get_registered_items():
    return sorted(
        [
            name
            for name in os.listdir(PROCESSED_DATA_DIR)
            if os.path.isdir(os.path.join(PROCESSED_DATA_DIR, name))
        ]
    )


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gerenciador de Itens")
        self.setGeometry(100, 100, 350, 500)

        self.layout = QVBoxLayout()

        self.label = QLabel("Itens já cadastrados:")
        self.layout.addWidget(self.label)

        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        self.add_button = QPushButton("Adicionar novo item")
        self.add_button.clicked.connect(self.add_new_item)
        self.layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Remover item selecionado")
        self.remove_button.clicked.connect(self.remove_selected_item)
        self.layout.addWidget(self.remove_button)

        self.train_button = QPushButton("Treinar novo modelo")
        self.train_button.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_button)

        self.detect_button = QPushButton("Iniciar detecção")
        self.detect_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.detect_button)

        self.setLayout(self.layout)
        self.populate_list()

        self.arduino_button = QPushButton("Iniciar Arduino")
        self.arduino_button.clicked.connect(self.start_arduino)
        self.layout.addWidget(self.arduino_button)

    def start_arduino(self):
        items = get_registered_items()
        if not items:
            QMessageBox.warning(self, "Sem itens", "Nenhum item encontrado.")
            return

        port_map = {}
        for item in items:
            pin, ok = QInputDialog.getText(
                self,
                f"Configurar pino para {item}",
                f"Qual pino do Arduino acende para '{item}' (entre 2 e 13)?",
            )
            if ok and pin.isdigit():
                pinAsInt = int(pin)
                if pinAsInt not in port_map and pinAsInt < 14 and pinAsInt > 1:
                    port_map[item] = pinAsInt
                else:
                    QMessageBox.warning(
                        self,
                        "Entrada inválida",
                        f"Pino inválido para '{item}', confira se ele está entre 2 e 13 e se ele já não foi utilizado.",
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Entrada inválida",
                    f"Pino inválido para '{item}'.",
                )
                return

        QMessageBox.information(self, "Sucesso", f"Pinos '{port_map}'.")

        port, ok = QInputDialog.getText(
            self,
            "Porta serial",
            "Informe a porta do Arduino (ex: /dev/ttyACM0 ou COM3):",
        )
        if not ok or not port:
            return

        try:
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as temp_file:
                json.dump(port_map, temp_file)
                temp_file_path = temp_file.name

            subprocess.run(
                [
                    sys.executable,
                    "src/arduino.py",
                    "--model",
                    get_latest_model(),
                    "--map",
                    temp_file_path,
                    "--port",
                    port,
                ],
                check=True,
            )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao iniciar: {str(e)}")

    def populate_list(self):
        self.list_widget.clear()
        items = get_registered_items()
        for item in items:
            QListWidgetItem(item, self.list_widget)

    def remove_selected_item(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "Nenhuma seleção", "Selecione um item para remover."
            )
            return

        item = selected_items[0].text()

        confirm = QMessageBox.question(
            self,
            "Confirmar remoção",
            f"Tem certeza que deseja remover '{item}'?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if confirm == QMessageBox.Yes:
            try:
                shutil.rmtree(os.path.join(PROCESSED_DATA_DIR, item))
                shutil.rmtree(os.path.join(CAPTURES_DIR, item))
                QMessageBox.information(self, "Sucesso", f"Item '{item}' removido.")
                self.populate_list()
            except Exception as e:
                QMessageBox.critical(
                    self, "Erro", f"Erro ao remover '{item}': {str(e)}"
                )

    def add_new_item(self):
        text, ok = QInputDialog.getText(self, "Novo item", "Nome do novo item:")
        if ok and text:
            try:
                subprocess.run([sys.executable, "src/capture.py", text], check=True)
                QMessageBox.information(
                    self, "Captura finalizada", f"Captura para '{text}' concluída."
                )
                self.populate_list()
            except subprocess.CalledProcessError:
                QMessageBox.critical(self, "Erro", "Erro ao capturar imagens.")

    def train_model(self):
        try:
            subprocess.run([sys.executable, "src/train_model.py"], check=True)
            QMessageBox.information(self, "Treinamento", "Modelo treinado com sucesso!")
        except subprocess.CalledProcessError:
            QMessageBox.critical(self, "Erro", "Erro ao treinar o modelo.")

    def start_detection(self):
        try:
            subprocess.run([sys.executable, "src/predict.py"], check=True)
        except subprocess.CalledProcessError:
            QMessageBox.critical(self, "Erro", "Erro ao iniciar a detecção.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
