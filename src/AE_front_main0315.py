import sys
import AE_front0318
from PyQt5.QtWidgets import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = QMainWindow()
    qianduanUi = AE_front0318.Ui_Form()
    qianduanUi.setupUi(win)
    win.show()
    sys.exit(app.exec_())

