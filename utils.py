import glob
import cv2
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations
import albumentations.pytorch
import itertools

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validation_sampler(root, n_data, n_val, csv):
    # assert ratio < 1, "ratio should be lower than 1"
    randper = torch.randperm(n_data).numpy()
    label_count = [0] * 5
    valid_idx = []
    train_idx = []
    for i in range(n_data):
        idx = randper[i]
        label = csv.iloc[idx, 1]
        if label_count[label] >= n_val / 5:
            train_idx.append(idx)
            continue
        valid_idx.append(idx)
        label_count[label] += 1

    return train_idx, valid_idx

class cassava_dataset_albu(torch.utils.data.Dataset):
    def __init__(self, root, csv=None, train_test="train", idx=None, transform=None):
        self.root = root
        self.csv = csv
        self.train_test = train_test
        self.index = idx
        self.transform = transform

        if self.train_test in ["train", "valid"]:
            assert idx is not None, "index list is None!"
            self.index = idx
            self.files = [f"{root}/{self.csv.iloc[i, 0]}" for i in self.index]
            self.csv = self.csv.iloc[self.index, :]
        elif self.train_test in ["test"]:
            self.files = glob.glob(f"{root}/*.jpg")
        
        self.basic_transform = albumentations.pytorch.ToTensorV2()
        self.transform = transform if transform is not None else self.basic_transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.csv.iloc[idx, 1]).long()

        # start = time.time()
        augmented = self.transform(image=image)
        image = augmented['image']
        # total_time = time.time() - start
        pack = (image, label)
        return pack

def run_train(x, y, net, optimizer):
    x = x.float().to(device)
    y = y.long().to(device)

    optimizer.zero_grad()
    pred = net.train()(x)
    loss = F.cross_entropy(pred, y, reduction='mean')
    with torch.no_grad():
        acc = (pred.argmax(dim=-1) == y).float().mean()
    loss.backward()
    optimizer.step()

    return loss, acc
    
def run_eval(x, y, net, optimizer):
    x = x.float().to(device)
    y = y.long().to(device)

    with torch.no_grad():
        pred = net.eval()(x)
        loss = F.cross_entropy(pred, y, reduction='mean')
        acc = (pred.argmax(dim=-1) == y).float().mean()
    return loss, acc

def run_epoch(dataset, dataloader, train, net, optimizer, epoch=None, writer=None, args=None):
    total_loss = 0.0
    total_acc = 0.0
    n_data = len(dataloader) * dataloader.batch_size 
    
    pbar = tqdm(total=n_data, position=0, leave=False)
    
    mode = 'Train' if train else 'Test'
    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(args.n_epochs))), args.n_epochs)
    
    for i, data in enumerate(dataloader):

        loss, acc = run_train(data[0], data[1], net, optimizer) if train else \
            run_eval(data[0], data[1], net, optimizer)
        '''
        if train and writer is not None:
            assert(epoch is not None)
            step = epoch * len(dataloader) + i
            writer.add_scalar('Loss/Train', loss, step)
            writer.add_scalar('Accuracy/Train', acc, step)
        '''
        batch_size = data[0].shape[0]
        total_loss += (loss * batch_size)
        total_acc += (acc * batch_size)

        pbar.set_description('{} {} Loss: {:f}, Acc : {:.4f}%'.format(
            epoch_str, mode, loss, acc))
        pbar.update(batch_size)


    mean_loss = total_loss / float(n_data)
    mean_acc = total_acc / float(n_data)
    return mean_loss, mean_acc

def run_epoch_train_and_test(
    train_dataset, train_dataloader, test_dataset, test_dataloader, net, 
    optimizer, args, epoch=None, writer=None):
    train_loss, train_acc = run_epoch(
        train_dataset, train_dataloader, train=args.train, net=net, optimizer=optimizer, epoch=epoch, writer=None, args=args)
    test_loss, test_acc = run_epoch(
        test_dataset, test_dataloader, train=False, net=net, optimizer=optimizer, epoch=epoch, writer=None, args=args)

    if writer is not None:

        assert(epoch is not None)
        step = (epoch + 1) * len(train_dataloader)
        writer.add_scalar('Loss/Train', train_loss, step)
        writer.add_scalar('Accuracy/Train', train_acc, step)

        writer.add_scalar('Loss/Test', test_loss, step)
        writer.add_scalar('Accuracy/Test', test_acc, step)

    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(args.n_epochs))), args.n_epochs)

    log = epoch_str + ' '
    log += 'Train Loss: {:f}, '.format(train_loss)
    log += 'Train Acc: {:.4f}%, '.format(train_acc)
    log += 'Test Loss: {:f}, '.format(test_loss)
    log += 'Test Acc: {:.4f}%.'.format(test_acc)
    print(log)
    return test_loss, test_acc

def get_all_preds(model, loader):
    with torch.no_grad():
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        all_preds = all_preds.to(device)
        all_labels = all_labels.to(device)
        for images, labels in loader:
            images = images.float().to(device)
            labels = labels.to(device)
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
    return all_preds, all_labels

def get_num_correct(preds, labels):
    return preds.argmax(dim=-1).eq(labels).sum().item()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')