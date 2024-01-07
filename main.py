import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from 窗口 import Ui_NERprogram
import json
from ner_model import BiLstm


def NER(s):
    global word2index, word2index, index2tag, index2word, model
    a = [word2index.get(i, word2index['<UNK>']) for i in s]
    word = torch.tensor(a, dtype=torch.int64, device='cuda:0').unsqueeze(0)
    tag_hat = model.forward(word)
    tag_hat = tag_hat.reshape(-1, tag_hat.shape[-1])
    word = word.reshape(-1)
    pre = tag_hat.argmax(1)
    res = ''
    for id, item in enumerate(pre):
        item = item.item()
        if item and index2tag[item] != '<PAD>':
            res += index2word[word[id].item()] + ' ' + index2tag[item] + '\n'
    return res


class MyMainWindow(QMainWindow, Ui_NERprogram):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

    def slot_button(self):
        s = self.textEdit.toPlainText()
        self.textBrowser.setText(NER(s))


def get_dict(name):
    with open('./dict/' + name + '.json', 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    word2index = get_dict('word2index')
    tag2index = get_dict('tag2index')
    index2word = get_dict('index2word')
    index2tag = get_dict('index2tag')

    index2tag = dict((int(key), index2tag[key]) for key in index2tag)
    index2word = dict((int(key), index2word[key]) for key in index2word)

    model = torch.load('bilstm.pth')
    model.eval()

    # 创建QApplication类的实例
    app = QApplication(sys.argv)
    # 创建一个窗口
    main_window = MyMainWindow()
    main_window.show()
    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())
