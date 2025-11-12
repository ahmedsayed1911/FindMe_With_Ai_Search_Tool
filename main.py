# ============================================================================
# Main 
# ============================================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from ui.feed_widget import FeedWidget
from ui.search_results_widget import SearchResultsWidget
from ui.search_widget import SearchWidget
from ui.add_post_widget import AddPostWidget
from chroma_manager import ChromaManager


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Missing Persons Finder - Admin")
        self.setStyleSheet("background-color: #1C1E21; color: white;")
        
        self.chroma_manager = ChromaManager()
        
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        
        self.feed = FeedWidget(self.chroma_manager)
        self.results_widget = SearchResultsWidget(self.feed.show_post_by_id)
        self.search = SearchWidget(self.results_widget, self.chroma_manager)
        self.add = AddPostWidget(self.feed_refresh, self.chroma_manager)
        
        left_layout.addWidget(self.search)
        left_layout.addWidget(self.results_widget)
        left_layout.addWidget(self.add)
        
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.feed)
        
        self.setLayout(main_layout)
    
    def feed_refresh(self):
        self.feed.refresh()


def main():
    app_qt = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    sys.exit(app_qt.exec_())


if __name__ == "__main__":
    main()