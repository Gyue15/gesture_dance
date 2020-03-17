import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPalette, QColor, QStandardItemModel, QStandardItem, QImage, QPixmap
from PyQt5.QtCore import QUrl, QDirIterator, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QFileDialog, QAction, QHBoxLayout, \
    QVBoxLayout, QSlider, QAbstractItemView, QHeaderView, QLabel
from PyQt5.QtMultimedia import QMediaPlaylist, QMediaPlayer, QMediaContent
import cv2

from hand_rec import HandDetection


class MusicApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.player = QMediaPlayer()
        self.playlist = QMediaPlaylist()
        self.title = 'GestureTunes'
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 340
        self.volume_initial_value = 60
        self.model = QStandardItemModel(0, 1)
        self.song_list = QtWidgets.QTableView()
        self.play_btn = QPushButton('播放')
        self.init_ui()
        self.image_to_show = None
        # 摄像头窗口
        self.widget = QWidget()
        self.widget.move(1000, 200)
        self.widget.setWindowTitle("手势窗口")  # 窗口标题
        self.videoFrame = QLabel('VideoCapture')
        video_area = QVBoxLayout()
        self.widget.setLayout(video_area)
        video_area.addWidget(self.videoFrame)
        self.widget.show()
        # 定时器
        self.timer = QTimer()
        self.timer.setInterval(20)
        self.timer.start()
        self.timer.timeout.connect(self.show_image)

    def init_ui(self):
        # Add file menu
        menubar = self.menuBar()
        filemenu = menubar.addMenu('File')
        windowmenu = menubar.addMenu('Window')

        fileAct = QAction('Open File', self)
        folderAct = QAction('Open Folder', self)
        themeAct = QAction('Toggle light/dark theme', self)

        filemenu.addAction(fileAct)
        filemenu.addAction(folderAct)
        windowmenu.addAction(themeAct)

        fileAct.triggered.connect(self.open_file)
        folderAct.triggered.connect(self.add_files)
        themeAct.triggered.connect(self.toggle_colors)

        self.add_listener()

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.toggle_colors()
        self.show()

    def add_listener(self):
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # 播放控件
        volume_slider = QSlider(Qt.Horizontal, self)
        volume_slider.setFocusPolicy(Qt.NoFocus)
        volume_slider.valueChanged[int].connect(self.change_volume)
        volume_slider.setValue(self.volume_initial_value)
        open_btn = QPushButton('打开...')
        prev_btn = QPushButton('上一首')
        next_btn = QPushButton('下一首')
        # 显示播放列表
        self.set_play_list()
        self.song_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.song_list.setShowGrid(False)
        self.song_list.setTextElideMode(QtCore.Qt.ElideLeft)

        # 按钮的layout
        control_area = QVBoxLayout()  # centralWidget
        media_controls = QHBoxLayout()
        # file_controls = QHBoxLayout()

        # 将播放控件添加到layout
        media_controls.addWidget(open_btn)
        media_controls.addWidget(prev_btn)
        media_controls.addWidget(self.play_btn)
        media_controls.addWidget(next_btn)
        media_controls.addWidget(volume_slider)

        # 将layout添加到界面中
        control_area.addWidget(self.song_list)
        control_area.addLayout(media_controls)
        wid.setLayout(control_area)

        # 设置监听
        self.play_btn.clicked.connect(self.start_or_stop)
        open_btn.clicked.connect(self.add_files)
        prev_btn.clicked.connect(self.prev)
        next_btn.clicked.connect(self.next)
        self.song_list.doubleClicked.connect(self.change_song)

        self.statusBar()
        self.playlist.currentMediaChanged.connect(self.song_changed)

    def set_play_list(self):
        self.model.clear()
        self.song_list.horizontalHeader().hide()
        # self.song_list.verticalHeader().hide()
        # QHeaderView()
        self.song_list.setModel(self.model)  # 把数据添加至QtableView中
        self.song_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.song_list.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置不可编辑单元格
        if self.playlist.mediaCount():  # 添加数据
            for index in range(self.playlist.mediaCount()):
                self.model.appendRow([  # 添加一行数据
                    QStandardItem(self.playlist.media(index).canonicalUrl().fileName())
                ])
            self.song_list.selectRow(0)

    def open_file(self):
        song = QFileDialog.getOpenFileName(self, "Open Song", "~", "Sound Files (*.mp3 *.ogg *.wav *.m4a)")

        if song[0] != '':
            url = QUrl.fromLocalFile(song[0])
            if self.playlist.mediaCount() == 0:
                self.playlist.addMedia(QMediaContent(url))
                self.player.setPlaylist(self.playlist)
                self.player.play()
            else:
                self.playlist.addMedia(QMediaContent(url))
        self.set_play_list()

    def add_files(self):
        if self.playlist.mediaCount() != 0:
            self.playlist.clear()
        self.folder_iterator()
        self.player.setPlaylist(self.playlist)
        self.player.playlist().setCurrentIndex(0)
        self.song_list.selectRow(0)
        self.set_play_list()
        self.player.pause()
        self.statusBar().showMessage("暂停：" + self.playlist.currentMedia().canonicalUrl().fileName())

    def folder_iterator(self):
        folder_chosen = QFileDialog.getExistingDirectory(self, 'Open Music Folder', '~')
        if folder_chosen is not None:
            it = QDirIterator(folder_chosen)
            it.next()
            while it.hasNext():
                if (not it.fileInfo().isDir()) and it.filePath() != '.':
                    fInfo = it.fileInfo()
                    if fInfo.suffix() in ('mp3', 'ogg', 'wav', 'm4a'):
                        self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(it.filePath())))
                it.next()
            if (not it.fileInfo().isDir()) and it.filePath() != '.':
                fInfo = it.fileInfo()
                if fInfo.suffix() in ('mp3', 'ogg', 'wav', 'm4a'):
                    self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(it.filePath())))

    def start_or_stop(self):
        if self.playlist.mediaCount() == 0:
            self.open_file()
        else:
            if self.player.state() == 1:  # 判断播放状态，1表示播放中，2表示暂停
                self.player.pause()
                self.play_btn.setText('播放')
                self.statusBar().showMessage("暂停：" + self.playlist.currentMedia().canonicalUrl().fileName())
            elif self.player.state() == 2:
                self.player.play()
                self.play_btn.setText('暂停')
                self.statusBar().showMessage("正在播放：" + self.playlist.currentMedia().canonicalUrl().fileName())

    def change_volume(self, value):
        self.player.setVolume(value)

    def prev(self):
        if self.playlist.mediaCount() == 0:
            self.open_file()
        elif self.playlist.mediaCount() != 0:
            self.player.playlist().previous()

    def shuffle(self):
        self.playlist.shuffle()

    def next(self):
        if self.playlist.mediaCount() == 0:
            self.open_file()
        elif self.playlist.mediaCount() != 0:
            self.player.playlist().next()

    def song_changed(self, media):
        if not media.isNull():
            url = media.canonicalUrl()
            self.statusBar().showMessage("正在播放：" + url.fileName())
            self.song_list.selectRow(self.playlist.currentIndex())

    def change_song(self):
        index = self.song_list.currentIndex().row()  # 获取双击所在行
        self.playlist.setCurrentIndex(index)

    def toggle_colors(self):
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(100, 149, 237))
        palette.setColor(QPalette.Highlight, QColor(100, 149, 237))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)

    def convert_image(self, frame):
        # height, width, bytes_per_component = frame.shape
        # bytes_per_line = bytes_per_component * width
        # # 变换彩色空间顺序
        # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        # # 转为QImage对象
        # image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # self.image_to_show = QPixmap.fromImage(image)
        if len(frame.shape) == 2:  # 若是灰度图则转为三通道
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
        # show_image(filename,rgb_image)
        # rgb_image=Image.open(filename)
        rgb_image = np.asanyarray(rgb_image)
        label_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)  # 转化为QImage
        self.image_to_show = QPixmap(label_image)
        # show_image("src resize image",image)

    def show_image(self):
        # print(self.image_to_show)
        if self.image_to_show is not None:
            self.videoFrame.setPixmap(self.image_to_show)


if __name__ == '__main__':
    # run_detection()
    app = QApplication(sys.argv)
    ex = MusicApp()
    detection_thread = HandDetection(ex)
    detection_thread.start()
    sys.exit(app.exec_())
