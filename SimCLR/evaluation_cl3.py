import torch
import sys
import numpy as np
import os
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets
import medmnist
from medmnist import INFO, Evaluator
from utils import *
from tqdm import trange
from base_model import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from collections import OrderedDict
import PIL
import torch.nn as nn
import torch.optim as optim
from utils_noise import *
def train(model, train_loader, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        one_hot = torch.nn.functional.one_hot(targets,num_classes=2)
        one_hot = torch.squeeze(one_hot,1)
        one_hot = one_hot.to(torch.float32).to(device)
        loss = criterion(outputs, one_hot)
        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1
        loss.backward()
        optimizer.step()
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss
def test(model, evaluator, data_loader, criterion, device, run, save_folder=None):

    model.eval()

    total_loss = []
    #print(torch.tensor([]).to(device))
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            #targets = targets.to(torch.float32).to(device)
            one_hot = torch.nn.functional.one_hot(targets,num_classes=2)
            one_hot = torch.squeeze(one_hot,1)
            one_hot = one_hot.to(torch.float32).to(device)
            loss = criterion(outputs, one_hot)
            m = nn.Sigmoid()
            outputs = m(outputs).to(device)
            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
data_flag = 'pneumoniamnist'
as_rgb = True
download = True
batch_size = 256
noise = 'gaussian'
resize= False
info = INFO[data_flag]
task = info['task']
n_channels = 3 if as_rgb else info['n_channels']
n_classes = len(info['label'])
prob = 0
DataClass = getattr(medmnist, info['python_class'])
if resize:
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])
else:
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])
if noise == 'none':
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)
elif noise =='jitter':
    noise_transform = inject_jitter(prob)
    train_dataset = DataClass(split='train', transform=noise_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=noise_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)
elif noise =='gray':
    noise_transform = inject_graynoise(prob)
    train_dataset = DataClass(split='train', transform=noise_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=noise_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)
elif noise =='flip':
    noise_transform = inject_randomflip(prob)
    train_dataset = DataClass(split='train', transform=noise_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=noise_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)
elif noise =='gaussian':
    noise_transform = inject_gaussiannoise()
    train_dataset = DataClass(split='train', transform=noise_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=noise_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)
else:
    pass
train_loader = data.DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=False)
val_loader = data.DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)




train_evaluator = medmnist.Evaluator(data_flag, 'train')
val_evaluator = medmnist.Evaluator(data_flag, 'val')
test_evaluator = medmnist.Evaluator(data_flag, 'test')
model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
checkpoint = torch.load('./embedding/checkpoint_0200.pth.tar', map_location=device)
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
    if k.startswith('backbone.'):
        if k.startswith('backbone') and not k.startswith('backbone.linear'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
    del state_dict[k]
log = model.load_state_dict(state_dict, strict=False)
for name, param in model.named_parameters():
    if name not in ['linear.weight', 'linear.bias']:
        param.requires_grad = False
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
run = 'model_cl'
output_root= './output'
num_epochs = 100
#train_metrics = test(model, train_evaluator, train_loader_at_eval,criterion, device, run, output_root)
#val_metrics = test(model, val_evaluator, val_loader,criterion, device, run, output_root)
#test_metrics = test(model, test_evaluator, test_loader,criterion, device, run, output_root)

'''print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
        'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
        'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))
'''
gamma=0.1
milestones = [0.5 * num_epochs, 0.75 * num_epochs]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

logs = ['loss', 'auc', 'acc']
train_logs = ['train_'+log for log in logs]
val_logs = ['val_'+log for log in logs]
test_logs = ['test_'+log for log in logs]
log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)

writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

best_auc = 0
best_epoch = 0
best_model = model

global iteration
iteration = 0

for epoch in trange(num_epochs):        
    train_loss = train(model, train_loader,criterion, optimizer, device, writer)
    #print("train finished")
    train_metrics = test(model, train_evaluator, train_loader_at_eval,criterion, device, run)
    val_metrics = test(model, val_evaluator, val_loader,criterion, device, run)
    test_metrics = test(model, test_evaluator, test_loader,criterion, device, run)
    
    scheduler.step()
    
    for i, key in enumerate(train_logs):
        log_dict[key] = train_metrics[i]
    for i, key in enumerate(val_logs):
        log_dict[key] = val_metrics[i]
    for i, key in enumerate(test_logs):
        log_dict[key] = test_metrics[i]

    for key, value in log_dict.items():
        writer.add_scalar(key, value, epoch)
        
    cur_auc = val_metrics[1]
    if cur_auc > best_auc:
        best_epoch = epoch
        best_auc = cur_auc
        best_model = model
        print('cur_best_auc:', best_auc)
        print('cur_best_epoch', best_epoch)

state = {
    'net': best_model.state_dict(),
}

path = os.path.join(output_root, 'best_model.pth')
torch.save(state, path)

train_metrics = test(best_model, train_evaluator, train_loader_at_eval,criterion, device, run, output_root)
val_metrics = test(best_model, val_evaluator, val_loader,criterion, device, run, output_root)
test_metrics = test(best_model, test_evaluator, test_loader,criterion, device, run, output_root)

train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

log = '%s\n' % (data_flag) + train_log + val_log + test_log
print(log)
        
with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
    f.write(log)  
        
writer.close()

