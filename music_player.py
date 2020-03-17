import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPalette, QColor, QStandardItemModel, QStandardItem, QImage, QPixmap, QIcon
from PyQt5.QtCore import QUrl, QDirIterator, Qt, QTimer, QCoreApplication
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
        self.title = 'GestureTunes - player'
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 340
        self.volume_initial_value = 60
        self.model = QStandardItemModel(0, 1)
        self.song_list = QtWidgets.QTableView()
        self.play_btn = QPushButton('播放')
        self.media_controls = QHBoxLayout()
        # 共享变量
        self.image_to_show = None
        self.rec_res = {"set": False, "used": True, "direction": None, "fingers": 0}
        # 摄像头窗口
        self.widget = QWidget()
        self.widget.move(1000, 200)
        self.widget.resize(300, 200)
        self.widget.setWindowTitle("GestureTunes - gesture panel")  # 窗口标题
        self.videoFrame = QLabel('正在打开摄像头，请稍等...')
        video_area = QVBoxLayout()
        self.widget.setLayout(video_area)
        video_area.addWidget(self.videoFrame)
        self.widget.show()
        # self.widget.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        # 定时器
        self.timer = QTimer()
        self.timer.setInterval(20)
        self.timer.start()
        self.timer.timeout.connect(self.show_image)
        self.have_song = False
        self.volume_slider = QSlider(Qt.Horizontal, self)
        self.volume_slider.setTracking(True)
        # 播放模式
        self.play_style = '列表循环'
        self.single_img = QIcon('pics/single.png')
        self.loop_img = QIcon('pics/loop.png')
        self.play_style_btn = QPushButton()
        self.play_style_btn.setIcon(self.loop_img)
        self.play_tip = ''

        self.playlist.setPlaybackMode(QMediaPlaylist.Loop)

        # 初始化界面
        self.init_ui()

    def init_ui(self):
        # Add file menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')

        fileAct = QAction('打开文件', self)
        folderAct = QAction('打开文件夹', self)

        file_menu.addAction(fileAct)
        file_menu.addAction(folderAct)

        fileAct.triggered.connect(self.open_file)
        folderAct.triggered.connect(self.add_files)

        self.add_listener()

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    def add_listener(self):
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # 播放控件
        self.volume_slider = QSlider(Qt.Horizontal, self)
        self.volume_slider.setFocusPolicy(Qt.NoFocus)
        self.volume_slider.valueChanged[int].connect(self.change_volume)
        self.volume_slider.setValue(self.volume_initial_value)
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
        self.media_controls = QHBoxLayout()

        # 将播放控件添加到layout
        self.media_controls.addWidget(open_btn)
        self.media_controls.addWidget(prev_btn)
        self.media_controls.addWidget(self.play_btn)
        self.media_controls.addWidget(next_btn)
        self.media_controls.addWidget(self.play_style_btn)
        self.media_controls.addWidget(self.volume_slider)

        # 将layout添加到界面中
        control_area.addWidget(self.song_list)
        control_area.addLayout(self.media_controls)
        wid.setLayout(control_area)

        # 设置监听
        self.play_btn.clicked.connect(self.start_or_stop)
        open_btn.clicked.connect(self.add_files)
        prev_btn.clicked.connect(self.prev)
        next_btn.clicked.connect(self.next)
        self.song_list.doubleClicked.connect(self.change_song)
        self.play_style_btn.clicked.connect(self.change_play_style)

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
        song = QFileDialog.getOpenFileName(self, "打开文件", "~", "音频文件 (*.mp3 *.ogg *.wav *.m4a)")

        if song[0] != '':
            url = QUrl.fromLocalFile(song[0])
            if self.playlist.mediaCount() == 0:
                self.playlist.addMedia(QMediaContent(url))
                self.player.setPlaylist(self.playlist)
                self.player.play()
            else:
                self.playlist.addMedia(QMediaContent(url))
        self.set_play_list()
        self.have_song = True

    def add_files(self):
        if self.playlist.mediaCount() != 0:
            self.playlist.clear()
        self.folder_iterator()
        self.player.setPlaylist(self.playlist)
        self.player.playlist().setCurrentIndex(0)
        self.song_list.selectRow(0)
        self.set_play_list()
        self.player.pause()
        self.play_tip = "暂停：" + self.playlist.currentMedia().canonicalUrl().fileName()
        self.statusBar().showMessage(self.play_tip + " - " + self.play_style)
        self.have_song = True

    def folder_iterator(self):
        folder_chosen = QFileDialog.getExistingDirectory(self, '打开文件夹', '~')
        if folder_chosen is not None:
            it = QDirIterator(folder_chosen)
            it.next()
            while it.hasNext():
                if (not it.fileInfo().isDir()) and it.filePath() != '.':
                    f_info = it.fileInfo()
                    if f_info.suffix() in ('mp3', 'ogg', 'wav', 'm4a'):
                        self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(it.filePath())))
                it.next()
            if (not it.fileInfo().isDir()) and it.filePath() != '.':
                f_info = it.fileInfo()
                if f_info.suffix() in ('mp3', 'ogg', 'wav', 'm4a'):
                    self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(it.filePath())))

    def start_or_stop(self):
        print(self.player.state())
        if self.playlist.mediaCount() == 0:
            self.open_file()
        else:
            if self.player.state() == QMediaPlayer.PlayingState or \
                    (self.player.state() == QMediaPlayer.PlayingState and self.play_btn.text() == '暂停'):
                self.player.pause()
                self.play_btn.setText('播放')
                self.play_tip = "暂停：" + self.playlist.currentMedia().canonicalUrl().fileName()
            elif self.player.state() == QMediaPlayer.PausedState or \
                    (self.player.state() == QMediaPlayer.PlayingState and self.play_btn.text() == '播放'):
                self.player.play()
                self.play_btn.setText('暂停')
                self.play_tip = "正在播放：" + self.playlist.currentMedia().canonicalUrl().fileName()
            self.statusBar().showMessage(self.play_tip + " - " + self.play_style)

    def change_volume(self, value):
        self.player.setVolume(value)

    def volume_up(self):
        volume = self.player.volume() + 10 if self.player.volume() + 10 <= 100 else 100
        self.player.setVolume(volume)
        self.volume_slider.setTracking(True)
        self.volume_slider.setValue(volume)

    def volume_down(self):
        volume = self.player.volume() - 10 if self.player.volume() - 10 >= 0 else 0
        self.player.setVolume(volume)
        self.volume_slider.setTracking(True)
        self.volume_slider.setValue(volume)

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
            self.play_tip = "正在播放：" + url.fileName()
            self.statusBar().showMessage(self.play_tip + " - " + self.play_style)
            self.song_list.selectRow(self.playlist.currentIndex())

    def change_song(self):
        index = self.song_list.currentIndex().row()  # 获取双击所在行
        self.playlist.setCurrentIndex(index)

    def change_play_style(self):
        if self.playlist.playbackMode() == QMediaPlaylist.Loop:
            self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)
            self.play_style = "单曲循环"
            self.play_style_btn.setIcon(self.single_img)
        elif self.playlist.playbackMode() == QMediaPlaylist.CurrentItemInLoop:
            self.playlist.setPlaybackMode(QMediaPlaylist.Loop)
            self.play_style = "列表循环"
            self.play_style_btn.setIcon(self.loop_img)
        self.statusBar().showMessage(self.play_tip + " - " + self.play_style)

    def convert_image(self, frame):
        if len(frame.shape) == 2:  # 若是灰度图则转为三通道
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
        rgb_image = np.asanyarray(rgb_image)
        label_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)  # 转化为QImage
        self.image_to_show = QPixmap(label_image)

    def show_image(self):
        if self.image_to_show is not None:
            self.videoFrame.setPixmap(self.image_to_show)
        if self.rec_res['set'] and not self.rec_res['used']:
            final_direction = self.rec_res['direction']
            final_fingers = self.rec_res['fingers']
            if final_fingers == 3:
                self.start_or_stop()
                self.rec_res['used'] = True
                return
            if final_fingers == 5:
                self.change_play_style()
                self.rec_res['used'] = True
                return
            if final_direction is not None and final_direction != 'NOT_FOUND':
                if final_direction == "RIGHT":
                    self.next()
                    self.rec_res['used'] = True
                if final_direction == "LEFT":
                    self.prev()
                    self.rec_res['used'] = True
                if final_direction == "UP":
                    self.volume_up()
                    self.rec_res['used'] = True
                if final_direction == "DOWN":
                    self.volume_down()
                    self.rec_res['used'] = True

    def set_rec_res(self, res):
        self.rec_res = res


def set_colors(music_app):
    music_app.setStyle("Fusion")
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
    music_app.setPalette(palette)
    pass


if __name__ == '__main__':
    # run_detection()
    app = QApplication(sys.argv)
    ex = MusicApp()
    set_colors(app)
    detection_thread = HandDetection(ex)
    detection_thread.start()
    sys.exit(app.exec_())
