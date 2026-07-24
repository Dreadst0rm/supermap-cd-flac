"""PySide6 Windows GUI: file upconverter + CD ripper."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QAction, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .io_audio import NATIVE_SUFFIXES, collect_audio_inputs
from .mb import AlbumMeta, lookup_disc, placeholder_meta
from .ml_upscaler import backend_name, ml_available
from .pipeline import RipOptions, rip_and_encode_track, upconvert_file
from .rip import DiscTOC, list_cd_drives, read_toc

HONESTY = (
    "Super Bit Mapping is encode-only. SuperMap synthesizes +16 bits inside "
    "each 16-bit bin (32-bit working precision), then noise-shaped quantizes "
    "to 20 or 24-bit FLAC. This does not recover a lost studio master. "
    "Lossy repair cleans codec artifacts and extends bandwidth from MP3/AAC — "
    "it is not true CD or studio FLAC. For archival purity, disable expand "
    "and skip lossy repair."
)

FILE_FILTER = (
    "Audio (*.flac *.wav *.ogg *.oga *.mp3 *.m4a *.aac *.opus *.aiff *.aif *.wv);;"
    "FLAC (*.flac);;WAV (*.wav);;Ogg (*.ogg *.oga);;All files (*.*)"
)

AUDIO_SUFFIXES = NATIVE_SUFFIXES | {
    ".mp3",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".wv",
    ".ape",
}


class RipWorker(QThread):
    log = Signal(str)
    progress = Signal(str, float)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(
        self,
        drive: str,
        toc: DiscTOC,
        meta: AlbumMeta,
        track_numbers: list[int],
        options: RipOptions,
    ) -> None:
        super().__init__()
        self.drive = drive
        self.toc = toc
        self.meta = meta
        self.track_numbers = track_numbers
        self.options = options

    def run(self) -> None:
        try:
            tracks = [t for t in self.toc.toc_audio() if t.number in self.track_numbers]
            for t in tracks:
                self.log.emit(f"Ripping track {t.number}...")

                def cb(msg: str, frac: float, _t=t) -> None:
                    self.progress.emit(msg, frac)

                result, paths = rip_and_encode_track(
                    self.drive, t, self.meta, self.options, progress=cb
                )
                self.log.emit(
                    f"Track {t.number}: CRC32={result.crc32:08X} "
                    f"AR_v1={result.accuraterip_v1:08X} "
                    f"AR_v2={result.accuraterip_v2:08X} "
                    f"verified_passes={result.verified_passes}"
                )
                for p in paths:
                    self.log.emit(f"  Wrote {p}")
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class UpconvertWorker(QThread):
    log = Signal(str)
    progress = Signal(str, float)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, paths: list[Path], options: RipOptions, force_ffmpeg: bool) -> None:
        super().__init__()
        self.paths = paths
        self.options = options
        self.force_ffmpeg = force_ffmpeg

    def run(self) -> None:
        try:
            total = len(self.paths)
            for i, src in enumerate(self.paths, start=1):
                self.log.emit(f"[{i}/{total}] Converting {src.name}...")

                def cb(msg: str, frac: float, _i=i, _src=src) -> None:
                    overall = ((_i - 1) + max(0.0, min(1.0, frac))) / total
                    self.progress.emit(f"{_src.name}: {msg}", overall)

                result, paths = upconvert_file(
                    src,
                    self.options,
                    force_ffmpeg=self.force_ffmpeg,
                    progress=cb,
                )
                # Land on this file's share of overall progress once at completion
                self.progress.emit(f"{src.name}: complete", i / total)
                self.log.emit(
                    f"  CRC32={result.crc32:08X} notes={';'.join(result.notes)}"
                )
                logged = set()
                for p in paths:
                    self.log.emit(f"  Wrote {p}")
                    sibling = p.parent / f"{p.stem}.convert.log"
                    # Main track log is without .16bit in the name
                    main_log = p.parent / f"{p.stem.replace('.16bit', '')}.convert.log"
                    for candidate in (sibling, main_log):
                        key = str(candidate.resolve()) if candidate.is_file() else None
                        if key and key not in logged:
                            logged.add(key)
                            self.log.emit(f"  Step log: {candidate}")
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class FileDropList(QListWidget):
    """List widget that accepts dragged audio files/folders."""

    files_dropped = Signal(list)  # list[Path]

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setMinimumHeight(220)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths: list[Path] = []
        for url in event.mimeData().urls():
            if isinstance(url, QUrl) and url.isLocalFile():
                paths.append(Path(url.toLocalFile()))
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class OptionsBar(QWidget):
    """Shared expand / output options."""

    def __init__(
        self,
        *,
        show_passes: bool = False,
        show_ffmpeg: bool = False,
        show_repair: bool = False,
    ) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row1 = QHBoxLayout()
        self.gap_fill_cb = QCheckBox("Expand 16+16->32 (SBM)")
        self.gap_fill_cb.setChecked(True)
        row1.addWidget(self.gap_fill_cb)
        row1.addWidget(QLabel("Out bits:"))
        self.bits_combo = QComboBox()
        self.bits_combo.addItem("24-bit FLAC", 24)
        self.bits_combo.addItem("20-bit (in 24-bit FLAC)", 20)
        row1.addWidget(self.bits_combo)
        self.ml_cb = QCheckBox("ML upscaler")
        hint = backend_name(prefer_torch=True)
        self.ml_cb.setToolTip(
            f"Backend: {hint}"
            + (" (torch)" if ml_available() else " (numpy spectral)")
        )
        row1.addWidget(self.ml_cb)
        self.sidecar_cb = QCheckBox("Keep 16-bit sidecar")
        row1.addWidget(self.sidecar_cb)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("FLAC level:"))
        self.level_spin = QSpinBox()
        self.level_spin.setRange(0, 8)
        self.level_spin.setValue(5)
        row2.addWidget(self.level_spin)
        self.passes_spin = QSpinBox()
        self.passes_spin.setRange(1, 4)
        self.passes_spin.setValue(2)
        if show_passes:
            row2.addWidget(QLabel("Rip passes:"))
            row2.addWidget(self.passes_spin)
        self.ffmpeg_cb = QCheckBox("Force ffmpeg decode")
        self.ffmpeg_cb.setToolTip("Decode via ffmpeg to s16le stereo @ 44.1 kHz")
        if show_ffmpeg:
            row2.addWidget(self.ffmpeg_cb)
        row2.addStretch(1)
        layout.addLayout(row2)

        self.repair_cb = QCheckBox("Repair lossy (MP3/AAC/etc.)")
        self.repair_cb.setToolTip(
            "Artifact cleanup + bandwidth extension before expand. "
            "Improves many poor encodes — not true CD/studio FLAC."
        )
        self.repair_strength = QComboBox()
        self.repair_strength.addItem("Light", "light")
        self.repair_strength.addItem("Medium", "medium")
        self.repair_strength.addItem("Strong", "strong")
        self.repair_strength.setCurrentIndex(1)
        if show_repair:
            row3 = QHBoxLayout()
            row3.addWidget(self.repair_cb)
            row3.addWidget(QLabel("Strength:"))
            row3.addWidget(self.repair_strength)
            row3.addStretch(1)
            layout.addLayout(row3)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self.out_label = QLabel(str(Path.cwd() / "rips"))
        self.out_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        out_row.addWidget(self.out_label, stretch=1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        out_row.addWidget(browse)
        layout.addLayout(out_row)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output folder", self.out_label.text())
        if path:
            self.out_label.setText(path)

    def options(self) -> RipOptions:
        return RipOptions(
            output_dir=Path(self.out_label.text()),
            gap_fill=self.gap_fill_cb.isChecked(),
            keep_16bit_sidecar=self.sidecar_cb.isChecked(),
            flac_level=self.level_spin.value(),
            passes=self.passes_spin.value(),
            output_bits=int(self.bits_combo.currentData()),
            ml_upscaler=self.ml_cb.isChecked(),
            repair_lossy=self.repair_cb.isChecked(),
            repair_strength=str(self.repair_strength.currentData() or "medium"),
        )


class FileConvertTab(QWidget):
    """Windows file load + convert panel."""

    log = Signal(str)
    progress = Signal(str, float)
    busy_changed = Signal(bool)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._paths: list[Path] = []
        self.worker: UpconvertWorker | None = None

        layout = QVBoxLayout(self)

        hint = QLabel(
            "Add FLAC, WAV, Ogg, MP3, AAC, or other audio — drag and drop files or folders. "
            "Enable Repair lossy for MP3/AAC cleanup + bandwidth extension."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.file_list = FileDropList()
        self.file_list.files_dropped.connect(self.add_paths)
        self.file_list.setToolTip("Drop audio files or folders here")
        layout.addWidget(self.file_list, stretch=1)

        btn_row = QHBoxLayout()
        add_files = QPushButton("Add files...")
        add_files.clicked.connect(self.add_files_dialog)
        btn_row.addWidget(add_files)
        add_folder = QPushButton("Add folder...")
        add_folder.clicked.connect(self.add_folder_dialog)
        btn_row.addWidget(add_folder)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self.remove_selected)
        btn_row.addWidget(remove)
        clear = QPushButton("Clear list")
        clear.clicked.connect(self.clear_list)
        btn_row.addWidget(clear)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        opts_box = QGroupBox("Convert options")
        opts_layout = QVBoxLayout(opts_box)
        self.options_bar = OptionsBar(show_ffmpeg=True, show_repair=True)
        opts_layout.addWidget(self.options_bar)
        layout.addWidget(opts_box)

        self.convert_btn = QPushButton("Convert to FLAC")
        self.convert_btn.setMinimumHeight(36)
        self.convert_btn.clicked.connect(self.start_convert)
        layout.addWidget(self.convert_btn)

        self.count_label = QLabel("0 files queued")
        layout.addWidget(self.count_label)

    def _refresh_list(self) -> None:
        self.file_list.clear()
        for p in self._paths:
            item = QListWidgetItem(str(p))
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.file_list.addItem(item)
        n = len(self._paths)
        self.count_label.setText(f"{n} file{'s' if n != 1 else ''} queued")
        self.convert_btn.setEnabled(n > 0 and self.worker is None)

    def add_paths(self, paths: list[Path]) -> None:
        try:
            found = collect_audio_inputs(paths, recursive=True)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "Missing path", str(exc))
            return
        # Also accept explicit files with known suffixes that collect might miss
        extra: list[Path] = []
        for p in paths:
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
                extra.append(p)
        merged = list(dict.fromkeys([*self._paths, *found, *extra]))
        self._paths = sorted(merged, key=lambda x: str(x).lower())
        self._refresh_list()
        self.log.emit(f"Queued {len(self._paths)} file(s)")

    def add_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select audio files", str(Path.cwd()), FILE_FILTER
        )
        if paths:
            self.add_paths([Path(p) for p in paths])

    def add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder", str(Path.cwd()))
        if folder:
            self.add_paths([Path(folder)])

    def remove_selected(self) -> None:
        selected = {
            Path(item.data(Qt.ItemDataRole.UserRole))
            for item in self.file_list.selectedItems()
        }
        self._paths = [p for p in self._paths if p not in selected]
        self._refresh_list()

    def clear_list(self) -> None:
        self._paths.clear()
        self._refresh_list()

    def start_convert(self) -> None:
        if not self._paths:
            QMessageBox.warning(self, "No files", "Add at least one audio file.")
            return
        options = self.options_bar.options()
        self.busy_changed.emit(True)
        self.convert_btn.setEnabled(False)
        # Reset single-completion gate on the main window via signal chain
        worker = UpconvertWorker(
            list(self._paths),
            options,
            force_ffmpeg=self.options_bar.ffmpeg_cb.isChecked(),
        )
        self.worker = worker
        worker.log.connect(self.log.emit)
        worker.progress.connect(self.progress.emit)
        worker.finished_ok.connect(self._on_ok)
        worker.failed.connect(self._on_fail)
        worker.start()

    def _on_ok(self) -> None:
        self.worker = None
        self.convert_btn.setEnabled(bool(self._paths))
        self.busy_changed.emit(False)
        self.finished_ok.emit("Convert complete")

    def _on_fail(self, err: str) -> None:
        self.worker = None
        self.convert_btn.setEnabled(bool(self._paths))
        self.busy_changed.emit(False)
        self.failed.emit(err)


class CdRipTab(QWidget):
    log = Signal(str)
    progress = Signal(str, float)
    busy_changed = Signal(bool)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.toc: DiscTOC | None = None
        self.meta: AlbumMeta | None = None
        self.worker: RipWorker | None = None

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Drive:"))
        self.drive_combo = QComboBox()
        row.addWidget(self.drive_combo, stretch=1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_drives)
        row.addWidget(refresh)
        self.read_btn = QPushButton("Read disc")
        self.read_btn.clicked.connect(self.read_disc)
        row.addWidget(self.read_btn)
        layout.addLayout(row)

        self.album_label = QLabel("No disc loaded.")
        layout.addWidget(self.album_label)

        self.track_list = QListWidget()
        self.track_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.track_list, stretch=1)

        opts_box = QGroupBox("Rip options")
        opts_layout = QVBoxLayout(opts_box)
        self.options_bar = OptionsBar(show_passes=True)
        opts_layout.addWidget(self.options_bar)
        layout.addWidget(opts_box)

        self.rip_btn = QPushButton("Rip selected tracks")
        self.rip_btn.setMinimumHeight(36)
        self.rip_btn.setEnabled(False)
        self.rip_btn.clicked.connect(self.start_rip)
        layout.addWidget(self.rip_btn)

        self.refresh_drives()

    def refresh_drives(self) -> None:
        self.drive_combo.clear()
        drives = list_cd_drives()
        if not drives:
            self.drive_combo.addItem("(no CD drives found)")
            self.read_btn.setEnabled(False)
        else:
            for d in drives:
                self.drive_combo.addItem(d)
            self.read_btn.setEnabled(True)

    def read_disc(self) -> None:
        drive = self.drive_combo.currentText()
        if not drive or drive.startswith("("):
            return
        try:
            self.toc = read_toc(drive)
        except Exception as exc:
            QMessageBox.critical(self, "TOC error", str(exc))
            return
        try:
            self.meta = lookup_disc(self.toc)
        except Exception as exc:
            self.log.emit(f"MusicBrainz lookup failed: {exc}")
            self.meta = placeholder_meta(self.toc)

        assert self.meta is not None
        self.album_label.setText(
            f"{self.meta.artist} - {self.meta.album}  (discid {self.meta.discid})"
        )
        self.track_list.clear()
        for t in self.toc.toc_audio():
            title = self.meta.title_for(t.number)
            item = QListWidgetItem(f"{t.number:02d}. {title}  ({t.duration_seconds:.1f}s)")
            item.setData(Qt.ItemDataRole.UserRole, t.number)
            item.setSelected(True)
            self.track_list.addItem(item)
        self.rip_btn.setEnabled(True)
        self.log.emit(f"Loaded TOC: {len(self.toc.toc_audio())} audio tracks")

    def start_rip(self) -> None:
        if not self.toc or not self.meta:
            return
        selected = [
            item.data(Qt.ItemDataRole.UserRole) for item in self.track_list.selectedItems()
        ]
        if not selected:
            QMessageBox.warning(self, "No tracks", "Select at least one track.")
            return
        options = self.options_bar.options()
        self.busy_changed.emit(True)
        self.rip_btn.setEnabled(False)
        worker = RipWorker(self.toc.drive, self.toc, self.meta, selected, options)
        self.worker = worker
        worker.log.connect(self.log.emit)
        worker.progress.connect(self.progress.emit)
        worker.finished_ok.connect(self._on_ok)
        worker.failed.connect(self._on_fail)
        worker.start()

    def _on_ok(self) -> None:
        self.worker = None
        self.rip_btn.setEnabled(self.toc is not None)
        self.busy_changed.emit(False)
        self.finished_ok.emit("Rip complete")

    def _on_fail(self, err: str) -> None:
        self.worker = None
        self.rip_btn.setEnabled(self.toc is not None)
        self.busy_changed.emit(False)
        self.failed.emit(err)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"SuperMap Converter  v{__version__}")
        self.resize(860, 680)
        self.setAcceptDrops(True)
        self._completion_shown = False

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.tabs = QTabWidget()
        self.file_tab = FileConvertTab()
        self.cd_tab = CdRipTab()
        self.tabs.addTab(self.file_tab, "Convert files")
        self.tabs.addTab(self.cd_tab, "Rip CD")
        layout.addWidget(self.tabs, stretch=2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        layout.addWidget(self.log)

        about_row = QHBoxLayout()
        about_row.addStretch(1)
        about = QPushButton("About")
        about.clicked.connect(self.show_about)
        about_row.addWidget(about)
        layout.addLayout(about_row)

        for tab in (self.file_tab, self.cd_tab):
            tab.log.connect(self.append_log)
            tab.progress.connect(self.on_progress)
            tab.finished_ok.connect(self.on_done)
            tab.failed.connect(self.on_fail)
            tab.busy_changed.connect(self._on_busy)

        self._make_menu()
        self.statusBar().showMessage("Ready — drop files on Convert files, or rip a CD")

    def _make_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        add = QAction("Add files...", self)
        add.triggered.connect(lambda: (self.tabs.setCurrentWidget(self.file_tab), self.file_tab.add_files_dialog()))
        menu.addAction(add)
        folder = QAction("Add folder...", self)
        folder.triggered.connect(
            lambda: (self.tabs.setCurrentWidget(self.file_tab), self.file_tab.add_folder_dialog())
        )
        menu.addAction(folder)
        menu.addSeparator()
        quit_act = QAction("Exit", self)
        quit_act.triggered.connect(self.close)
        menu.addAction(quit_act)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def append_log(self, text: str) -> None:
        self.log.append(text)

    def _on_busy(self, busy: bool) -> None:
        self.tabs.setEnabled(not busy)
        if busy:
            self._completion_shown = False
            self.progress.setValue(0)

    def on_progress(self, msg: str, frac: float) -> None:
        pct = int(max(0.0, min(1.0, frac)) * 100)
        self.progress.setValue(pct)
        self.statusBar().showMessage(msg)

    def on_done(self, title: str = "Done") -> None:
        # Complete only once when the batch finishes at 100%
        if self._completion_shown:
            return
        self._completion_shown = True
        self.progress.setValue(100)
        self.append_log("Done.")
        self.statusBar().showMessage(title)
        QMessageBox.information(self, title, "Finished successfully.")

    def on_fail(self, err: str) -> None:
        self.append_log(err)
        self.statusBar().showMessage("Failed")
        QMessageBox.critical(self, "Failed", err.splitlines()[0])

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About SuperMap Converter",
            f"<b>SuperMap Converter</b> v{__version__}<br><br>"
            "Load 16-bit / 44.1 audio (FLAC, WAV, Ogg, or via ffmpeg) and "
            "convert with SBM-style expand to 20/24-bit FLAC.<br><br>"
            f"{HONESTY}",
        )

    # Allow dropping on the main window while Convert tab is active
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.tabs.setCurrentWidget(self.file_tab)
            self.file_tab.add_paths(paths)
            event.acceptProposedAction()


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SuperMap Converter")
    app.setOrganizationName("SuperMap")
    win = MainWindow()
    win.show()
    return app.exec()


def run_gui_main() -> None:
    """Console-script entry point for supermap-cd-gui."""
    raise SystemExit(run_gui())
