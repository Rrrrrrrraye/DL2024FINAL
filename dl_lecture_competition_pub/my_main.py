import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast, GradScaler

# load the pretrained model
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
bert_model = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        # using TinyBERT's tokenizer
        self.df['question_tokens'] = self.df['question'].apply(lambda x: tokenizer(x, padding='max_length', max_length=64, truncation=True, return_tensors='pt'))

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            self.answer2idx = {}
            self.idx2answer = {}
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question_tokens : dict
            質問文をトークン化したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}").convert('RGB')
        image = self.transform(image)
        question_tokens = {k: v.squeeze(0) for k, v in self.df['question_tokens'][idx].items()}

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, question_tokens, torch.tensor(answers), int(mode_answer_idx)

        else:
            return image, question_tokens

    def __len__(self):
        return len(self.df)

# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

# 3. モデルのの実装
class VQAModel(nn.Module):
    def __init__(self, bert_model, n_answer: int):
        super().__init__()
        # 使用预训练的ResNet50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity() 
        self.bert = bert_model
        self.fc = nn.Sequential(
            nn.Linear(2048 + 312, 512),  # ResNet50 output vim：2048，TinyBERT output vim:312
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question_tokens):
        image_feature = self.resnet(image)  # 画像の特徴量
        outputs = self.bert(**question_tokens)
        question_feature = outputs.last_hidden_state[:, 0, :]
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        return x

# 4. 学習の実装
scaler = GradScaler()

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question_tokens, answers, mode_answer in dataloader:
        image, answers, mode_answer = image.to(device), answers.to(device), mode_answer.to(device)
        question_tokens = {k: v.squeeze(1).to(device) for k, v in question_tokens.items()}

        optimizer.zero_grad()
        with autocast():
            pred = model(image, question_tokens)
            loss = criterion(pred, mode_answer)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question_tokens, answers, mode_answer in dataloader:
        image, answers, mode_answer = image.to(device), answers.to(device), mode_answer.to(device)
        question_tokens = {k: v.to(device) for k, v in question_tokens.items()}

        pred = model(image, question_tokens)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = VQAModel(bert_model, n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    num_epoch = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5) 

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question_tokens in test_loader:
        image = image.to(device)
        question_tokens = {k: v.to(device) for k, v in question_tokens.items()}
        pred = model(image, question_tokens)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
