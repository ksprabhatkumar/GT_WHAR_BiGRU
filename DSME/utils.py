import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import random
import time
import os
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import torch
from torch.optim import Adam
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader, random_split
import warnings
warnings.filterwarnings('ignore')
from torchmetrics import MetricCollection
class MyDataset(Dataset):
    def __init__(self,  dataset_name: str, sliding_window_length: int, volunteer_list=None, relative_path: str = "../har_data/preprocessed/"):
        path = relative_path + dataset_name + "/"
        self.x, self.y, self.p = load_by_volunteers(volunteer_list, path, sliding_window_length)
        self.x, self.y, self.p = torch.from_numpy(self.x), torch.from_numpy(self.y), torch.from_numpy(self.p)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.p[index]

class MyLoss(nn.Module):
        def __init__(self, alpha=0.5, beta=0.5) -> None:
            super().__init__()
            self.model_loss = nn.CrossEntropyLoss()
            self.moe_loss = nn.CrossEntropyLoss()
            self.weights_loss = nn.MSELoss()
            self.alpha = alpha
            self.beta = beta
        def forward(self, y_pred_model, y_pred_moe, y_true, weights=None):
            model_loss = self.model_loss(y_pred_model, y_true)
            moe_loss = self.moe_loss(y_pred_moe, y_true)
            total_loss = self.alpha * model_loss + (1 - self.alpha) * (moe_loss)
            if weights is not None:
                num = weights.shape[1]
                equal_weights = np.zeros(weights.detach().cpu().numpy().shape)
                equal_weights.fill(1.0/num)
                equal_weights = torch.tensor(equal_weights, dtype=torch.float32).cuda()
                equal_loss = self.weights_loss(weights, equal_weights)
                total_loss += self.beta * equal_loss
            return total_loss, model_loss, moe_loss, equal_loss
        
        
def load_by_volunteers(volunteer_list, path, sliding_window_length):
    X,y,p = None, None, None
    path = path + f"{sliding_window_length}/"
    for i in range(len(volunteer_list)):
        try:
            idx = volunteer_list[i]
            if i == 0:
                X = np.load(path + f"domain{idx}_x.npy")
                y = np.load(path + f"domain{idx}_y.npy")
                p = np.empty(y.shape)
                p.fill(idx)
                continue
            temp_X = np.load(path + f"domain{idx}_x.npy")
            temp_y = np.load(path + f"domain{idx}_y.npy")
            temp_p = np.empty(temp_y.shape)
            temp_p.fill(idx)
            X = np.concatenate([X, temp_X])
            y = np.concatenate([y, temp_y])
            p = np.concatenate([p, temp_p])
        except FileNotFoundError:
            print(f"找不到{path}文件夹!!!!!")
            exit(1)
    return X, y, p

def get_optimizer(model, args):
    return Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

def get_loss(args):
    if args.criterion == 'ce':
        return nn.CrossEntropyLoss()
    if args.criterion == 'gile':
        raise NotImplementedError
    if args.criterion == 'domain':
        return MyLoss(args.alpha)
    
def get_model(args):
    model_name = args.model_name
    num_classes = args.num_classes
    sliding_window_length = args.sliding_window_length
    num_channels = args.num_channels
    model = None
    if model_name == 'deepconvlstm':
        from model.DeepConvLSTM import DeepConvLSTM
        model = DeepConvLSTM(num_classes,
                             num_channels,
                             32,
                             64)
    elif model_name == 'transformer':
        from model.VisionTransformer import VisionTransformer
        model = VisionTransformer(num_classes,num_channels,
                                                    sliding_window_length, num_blocks=3, nb_head=4, 
                                                    hidden_dim=8,
                                                    dropout=0.3,args=args)
    elif model_name == 'mtsdnet':
        from model.MTSDNet import MTSDNet

        model = MTSDNet(out_channel=num_classes,in_channel=num_channels,length=sliding_window_length, hidden=128, dim=4, structure_str="tsg")
    elif model_name == "ddnn":
        from model.DDNN import DDNN
        model = DDNN(128, n_lstm_layer=1,d_AE=50, sliding_window_length=sliding_window_length, num_channels=num_channels,num_classes=num_classes)
    elif model_name == "gile":
        from model.GILE import GILE
        model = GILE(args)
    elif model_name == "tripleattention":
        from model.TripletAttentionCNN import TripleAttentionCNN,TripleAttentionResnet
        model = TripleAttentionResnet(args.sliding_window_length, args.num_channels, args.num_classes, args.out_channels)
    elif model_name == 'resnet':
        from model.ResNet import Resnet
        model = Resnet(args.sliding_window_length, args.num_channels, args.num_classes, args.out_channels)
    elif model_name == 'cnn':
        from model.CNN import CNN
        model = CNN(args.sliding_window_length, args.num_channels, args.num_classes, args.out_channels)
    elif model_name == 'resnetmeta':
        from model.ResNetMeta import ResnetMeta
        model = ResnetMeta(args.sliding_window_length, args.num_channels, args.num_classes, args.out_channels)
    elif model_name == 'cnnmeta':
        from model.CNNMeta import CNNMeta
        model = CNNMeta(args.sliding_window_length, args.num_channels, args.num_classes, args.out_channels)
    elif model_name == "dualattention":
        from model.DualAttentionCNN import DualAttentionCNN,DualAttentionResnet
        model = DualAttentionResnet(args.sliding_window_length, args.num_channels, args.num_classes, args.out_channels * 2)
    elif model_name == 'deepcoral':
        from model.DeepCoral import DeepCoralCNN
        model = DeepCoralCNN(args.sliding_window_length, args.num_channels, args.num_classes, args.out_channels)
    elif model_name == 'dann':
        from model.DANNCNN import DANNCNN
        model = DANNCNN(args.sliding_window_length, args.num_channels, args.num_classes ,args.out_channels)
    elif model_name == 'rsc':
        from model.RSC import RSCCNN
        model = RSCCNN(args.sliding_window_length, args.num_channels, args.num_classes, args.out_channels)
    elif model_name == 'tccsnet':
        from model.TCCSNet import CSNet
        model = CSNet(args.sliding_window_length, args.num_channels, args.num_classes, args.cuda_device)
    elif model_name == 'gruinc':
        from model.GRUINC import GRUINC
        model = GRUINC(args.sliding_window_length, args.num_channels, args.num_classes, filter=32)
    elif model_name == 'elk':
        from model.ELKCNN import ELK_CNN
        model = ELK_CNN(args.num_channels, args.num_classes)
    return model
def get_model_info(args):
    import warnings
    warnings.warn("模型参数显示已废弃，不要用", DeprecationWarning)
    print(f"=========================================={args.model_name}===================================")   
    model = get_model(args) 
    from torchinfo import summary
    summary(model=model.cuda(), 
            input_data=(torch.rand(1,args.sliding_window_length, args.num_channels)), 
            device='cuda:0',batch_dim=0)
    print(f"=========================================={args.model_name}===================================")
    return  
    
    
def get_time() -> str:
    return time.strftime('%Y/%m/%d/%H:%M', time.localtime())

def get_metrics(args):
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision, Accuracy
    metrics = MetricCollection({
        "acc": MulticlassAccuracy(num_classes=args.num_classes, average="weighted"),
        "acc_mi": MulticlassAccuracy(num_classes=args.num_classes,average='micro'),
        # "acc_w": MulticlassAccuracy(num_classes=args.num_classes, average='weighted'),
        "wf1": MulticlassF1Score(num_classes=args.num_classes, average="weighted"),
        "maf": MulticlassF1Score(num_classes=args.num_classes, average="macro"),
        "pre": MulticlassPrecision(num_classes=args.num_classes, average="macro"),
        "rec": MulticlassRecall(num_classes=args.num_classes, average="macro"),
        
    })
    return metrics
    

def get_result_name(args):
    now = get_time()
    result_path = f'{args.save_path}/{args.dataset_name}'
    result_filename = f'{args.epochs}_{args.batch_size}_{args.sliding_window_length}_{args.model_name}{"" if args.use_moe == 0 else "_moe"}.csv'
    result_name = f"{result_path}/{now}/{result_filename}"
    # make dir and make empty file
    os.makedirs(f"{result_path}/{now}", exist_ok=True)
    
    return result_name, result_path

def save_by_result_name(result_path, res, args):
    if args.use_moe == 1:
        res = str(res)[1:-1]
    else:
        res = str(res[:-1])[1:-1]
    with open(f'{result_path}/{args.model_name}{"_moe" if args.use_moe == 1 else ""}_best_result.csv', 'a') as f:
        f.write(f"{res}, {args.volunteer_split}\n")

def get_loader_by_volunteers(args):
    volunteer_list = args.volunteer_split.split("|")
    res_loader = []
    for i in range(len(volunteer_list)):
        temp = volunteer_list[i].split(",")
        dataset = MyDataset(dataset_name=args.dataset_name,sliding_window_length=args.sliding_window_length,volunteer_list=temp)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,num_workers=4)
        print(f'loader #{i+1} batch:{len(loader)}')
        res_loader.append(loader)
    return res_loader

def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def dataset_preprocess(args):
    import sys
    sys.path.append("./data/")
    print(f"{args.dataset_name}: length={args.sliding_window_length}, step={args.sliding_window_step}")
    from data.prerprocess import main
    main(args.sliding_window_length, args.sliding_window_step, args.dataset_name)    

def weights_init(model):
    for m in model.modules():
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.LSTM)):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        if isinstance(model, (torch.nn.ConvTranspose2d, torch.nn.MaxUnpool2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in")
            

def train_test_with_torchmetrics(model:nn.Module, train_loader, test_loader, result_name, result_path, args):
    
    print(f"==============================================training {args.model_name}==============================================")
    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    model = model.cuda()
    metrics = get_metrics(args).cuda()
    # ============================================================
    if args.model_name == 'gile':
        false_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ============================================================
    criterion = get_loss(args)
    optimizer = get_optimizer(model, args)
    total_batches = len(train_loader)
    epochs = args.epochs
    
    
    
    weights_init(model)
    scheduler = StepLR(optimizer=optimizer,step_size=20 )
    best_acc = 0.0
    f = open(result_name, mode='w+')
    f.write("epoch,train_acc,train_wf1,test_acc,test_wf1,train_loss\n")
    f.close()
    for epoch in range(epochs):  # loop over the dataset multiple times
        metrics.reset()
        result = []
        total_loss = 0
        model.train()
        tqdm_train_loader = tqdm((train_loader), total=(total_batches))
        for train_data in tqdm_train_loader:        
            train_X, train_y, train_p = train_data
            train_X, train_y, train_p = train_X.float().cuda(), train_y.long().cuda(), train_p.long().cuda()
            train_y = train_y.squeeze(1) # [X, 1] > [X,]
            train_p = train_p.squeeze(1) - 1 # 来自数据源的域标签是1开始
            # ============================================================
            if args.model_name == 'gile':
                optimizer.zero_grad()
                false_optimizer.zero_grad()
                loss, _ = model.loss_function(train_p, train_X, train_y)
                loss_false = model.loss_function_false(train_p, train_X, train_y)
                loss.backward()
                loss_false.backward()
                optimizer.step()
                false_optimizer.step()
                predict_y= model.classifier(train_X)[1] 
            # ============================================================
            else:
                optimizer.zero_grad()
                predict_y = model(train_X)
                loss = criterion(predict_y, train_y)
                loss.backward()
                optimizer.step()    
            total_loss += loss.item()
            metrics.update(predict_y, train_y)
            tqdm_train_loader.set_description(f'Epoch [{epoch+1}/{epochs}]')
            tqdm_train_loader.set_postfix_str(f'loss:{loss.item():.4f}')
        scheduler.step()    
        accAndF1 = metrics.compute()    
        print(f'Train: accuracy: {accAndF1["acc"].item() * 100:.2f}%, weighted_f1: {accAndF1["wf1"].item() * 100:.2f}%, loss: {total_loss / total_batches:.4f}')
        # Test time!
        result.extend([epoch+1, round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2)])
        model.eval()
        with torch.no_grad():       
            metrics.reset()
            for test_data in test_loader:              
                test_X, test_y, _  = test_data
                test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
                test_y = test_y.squeeze(1) # [X, 1] > [X,]
                # ============================================================
                if args.model_name == "gile":
                    predict_y = model.classifier(test_X)[1]
                # ============================================================
                else:
                    predict_y = model(test_X) #  超大失误，test_X写成train_X
                metrics.update(predict_y, test_y)
        # 本轮测试的所有结果 
        accAndF1 = metrics.compute()
        print(f'Test: accuracy: {accAndF1["acc"] * 100:.2f}%, weighted_f1: {accAndF1["wf1"] * 100:.2f}%')
        result.extend([round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2), round(total_loss / total_batches , 4)])
        if True:
            best_acc = accAndF1["acc"]
            best_wf1 = accAndF1["wf1"]
            best_model = model
            best_metrics = accAndF1
            best_result = result
        with open(result_name, mode='a') as f:
            f.write(str(result)[1:-1] + "\n")
    torch.save({"model":best_model.state_dict()}.update(best_metrics), f"{result_path}/{args.model_name}_{args.volunteer_split}_acc={best_acc * 100:.2f}_wf1={best_wf1 * 100:.2f}_seed={args.seed}.ckpt") 
    return best_result

def merge(train_loaders, args):
    """从多个DataLoader中取出batch，然后把这些堆叠多组batch

    Args:
        train_loaders: _训练集DataLoader列表_
        args: _命令行字符串_

    Returns:
        _堆叠好的固定batch大小的特征、标签、编号_
    """
    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    res_x = torch.empty(0, args.batch_size, args.sliding_window_length, args.num_channels).cuda()
    res_y = torch.empty(0, args.batch_size, 1).cuda()
    res_p = torch.empty(0, args.batch_size, 1).cuda()
    for train_loader in train_loaders:
        
        for batch_index, data in enumerate(train_loader):
            x, y, p = data
            x, y, p = x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda(), p.unsqueeze(0).cuda()
            
            res_x = torch.cat([res_x, x], dim=0)
            res_y = torch.cat([res_y, y], dim=0)
            res_p = torch.cat([res_p, p], dim=0)  
    return (res_x, res_y, res_p)


def train_test_moe_end2end_with_torchmetrics(models, moe, train_loaders, test_loader, result_name , result_path,args): 
    print(f"==============================================training {args.model_name}_moe==============================================")
    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    params, lr = [], args.lr # 学习率
    for model in models: # [{'params': a.parameters},{},...]
        params.append({"params":model.parameters()})
    params.append({"params":moe.parameters()})
    optimizer = Adam(params=params, lr=lr) # 一个优化器调整基模型+moe的参数
    scheduler = StepLR(optimizer=optimizer,step_size=20)
    criterion = MyLoss(alpha=args.alpha, beta=args.beta) # 
    metrics = get_metrics(args).cuda()
    weights_init(moe)
    moe.cuda()
    for i in range(len(models)):
        weights_init(models[i])
        models[i].cuda()
    best_acc = 0.0
    # 志愿者标签映射到模型编号
    volunteer_2_submodel_idx = {}
    temp_idx = args.volunteer_split.split("|")
    for j in range(len(temp_idx)):
        volunteer_2_submodel_idx.update({int(temp_idx[j]):j})
    train_loader = merge(train_loaders, args) # 多个loader按batch维度融合

    total_batches = train_loader[0].shape[0]
    # 训练开始
    all_loss, all_moe_loss, all_domain_loss, all_equal_loss= [],[],[], []
    all_acc = []
    for epoch in range(args.epochs):
        result = []
        metrics.reset()
        total_weights = torch.empty(0, len(models), requires_grad=False).cuda() # 所有权重
        # (x, y, p) = train_loader
        total_index = 0
        total_loss = 0.0
        total_moe_loss, total_equal_loss = 0.0, 0.0
        total_domains_loss = 0.0
        random_batches = torch.randperm(total_batches) # 按batch打乱，获得随机索引
        tqdm_random_batches = tqdm(random_batches, total=len(random_batches))
        
        for batch_index in tqdm_random_batches: 
            # 假设一个batch里的志愿者标签一样
            batch_index = batch_index.item() # tensor->int
            total_index += 1 # 已经训练的batch数量
            # 从原数据取出一个batch
            train_X, train_y, train_p = train_loader[0][batch_index],train_loader[1][batch_index],train_loader[2][batch_index]
            train_p = volunteer_2_submodel_idx[int(train_p[0].long().item())] # 数组下标1开始的志愿者编号对应的基模型下标
            train_X, train_y = train_X.float().cuda(), train_y.long().cuda()
            train_y = train_y.squeeze(1) # [X, 1] > [X,]
        
            optimizer.zero_grad()
             # 计算moe权重
            if int(args.equal) == 1:
                weights = torch.tensor(np.array([1.0/(args.num_volunteers-1) for _ in range(args.num_volunteers-1)])).cuda().reshape(1,-1)
            else:
                weights = moe(train_X)
            total_weights = torch.cat([total_weights, weights], dim=0) # 汇总权重
            for j in range(len(models)):
                models[j].eval() # 挨个冻结基模型参数
            models[train_p].train() # 解冻对应志愿者的基模型参数，这里需要映射            
            # 计算moe合成结果
            moe_predict_y = torch.zeros(args.batch_size, args.num_classes).cuda()
            model_p_predict_y = None
            for j in range(len(models)):
                if int(args.equal) == 0:
                    temp = weights[:, j].unsqueeze(1)
                else:
                    temp = torch.ones(1,1).cuda()
                model_predict_y = models[j](train_X)
                if j == train_p: # 将对应基模型的预测结果加入总列表
                    model_p_predict_y = model_predict_y
                moe_predict_y += torch.mul(temp, model_predict_y)
            metrics.update(moe_predict_y, train_y)
            
            loss, model_loss, moe_loss, equal_loss = criterion(model_p_predict_y, moe_predict_y, train_y, weights) 
            loss.backward()
            total_loss += loss.item()
            total_domains_loss+=model_loss.item()
            total_moe_loss += moe_loss.item()
            total_equal_loss = equal_loss.item()
            optimizer.step()
            tqdm_random_batches.set_description(f'Epoch [{epoch+1}/{args.epochs}] Loader [{total_index}/{len(random_batches)}] ')
            tqdm_random_batches.set_postfix_str(f"loss:{loss.item():.4f}")
        scheduler.step()
        accAndF1 = metrics.compute()
        result.extend([epoch+1, round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2)])
        print(f'Train: accuracy: {accAndF1["acc"].item() * 100:.2f}%, weighted_f1: {accAndF1["wf1"]* 100:.2f}%, loss: {total_loss / total_batches:.4f}')
        print(f'Train: loss total loss({total_loss / total_batches:.4f}) = alpha ({args.alpha})  x domain loss ({total_domains_loss / total_batches:.4f}) + (1-alpha) ({1-args.alpha}) x moe loss ({total_moe_loss / total_batches:.4f}) + beta ({args.beta}) x equal loss ({total_equal_loss / total_batches:.4f})')
        print(f"Train: weights: {np.round(torch.mean(total_weights, dim=0).detach().cpu().numpy(), 2)}")
        
        all_loss.append(total_loss / total_batches)
        all_equal_loss.append(total_equal_loss / total_batches)
        all_moe_loss.append(total_moe_loss/ total_batches)
        all_acc.append(accAndF1["acc"].item())
                
        all_domain_loss.append(total_domains_loss/total_batches)
        # 验证
        for j in range(len(models)):
            models[j].eval() # 冻结所有模型参数   
        moe.eval()
        with torch.no_grad():
            metrics.reset()
            total_weights = torch.empty(0, len(models), requires_grad=False) # 所有权重
            for data in test_loader: 
                # 从原数据取出一个batch
                test_X, test_y, _ = data
                test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
                test_y = test_y.squeeze(1) # [X, 1] > [X,]
                
                if int(args.equal) == 1:
                    weights = torch.tensor(np.array([1.0/(args.num_volunteers-1) for _ in range(args.num_volunteers-1)])).cuda().reshape(1,-1)
                else:
                    weights = moe(train_X)
                total_weights = torch.cat([total_weights, weights.cpu()], dim=0) # 汇总权重          
                # 计算moe合成结果
                moe_predict_y = torch.zeros(args.batch_size, args.num_classes).cuda()
                for j in range(len(models)):
                    if int(args.equal) == 0:
                        temp = weights[:, j].unsqueeze(1)
                    else:
                        temp = torch.ones(1,1).cuda()
                    model_predict_y = models[j](test_X) # 每个基模型对数据的预测
                    moe_predict_y += torch.mul(temp, model_predict_y)
                metrics.update(moe_predict_y, test_y)
            accAndF1 = metrics.compute()
            print(f'Test: accuracy: {accAndF1["acc"] * 100:.2f}%, weighted_f1: {accAndF1["wf1"] * 100:.2f}%')
            print(f"Test: weights: {np.round(torch.mean(total_weights, dim=0).detach().cpu().numpy(), 2)}")
            result.extend([round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2), round(total_loss / total_batches , 4)])
            # 如果是最佳则保存
            if best_acc < accAndF1["acc"]:
                best_acc = accAndF1["acc"]
                best_wf1 = accAndF1["wf1"]
                best_models = models
                best_moe = moe
                best_metrics = accAndF1
                
                best_weights = np.round(torch.mean(total_weights, dim=0).numpy(),2)
                result.append(best_weights)
                best_result = result
                
            with open(result_name, mode='a') as f:
                f.write(str(result)[1:-1] + "\n")
    ckpts = {}
    for i in range(len(best_models)):
        ckpts.update({f"{args.model_name}_{i}": best_models[i].state_dict()})
    ckpts.update(best_metrics)
    ckpts.update({"moe":best_moe.state_dict(), "weight":best_weights})
    torch.save(ckpts, f"{result_path}/{args.model_name}_moe_{args.volunteer_split}_acc={best_acc * 100:.2f}_wf1={best_wf1 * 100:.2f}_seed={args.seed}.ckpt") 
    if args.return_loss == 1:
        return all_loss, all_domain_loss, all_moe_loss, all_equal_loss, all_acc
    return best_result


    print(f"==============================================training {args.model_name}_moe==============================================")
    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    metrics = get_metrics(args).cuda()
    best_acc = 0.0
    expert_path = f"{result_path}/experts/{args.dataset_name}"
    import os 
    os.makedirs(expert_path, exist_ok=True)
    
    # train submodels
    for i in range(len(train_loaders)):
        print(f"-----------------------------training {args.model_name}_{i}-----------------------------")
        model = get_model(args)
        trained_model = train_with_torchmetrics(model, train_loaders[i], result_name,args)
        torch.save(trained_model, f"{expert_path}/{args.model_name}_{i}.pt")
        del trained_model
    # moe setup
    models = [torch.load(f"{expert_path}/{args.model_name}_{i}.pt") for i in range(len(train_loaders))]
    for i in range(len(models)):
        models[i] = models[i].eval()
    optimizer = Adam(params=moe.parameters(), lr=args.lr) # 一个优化器调整基模型+moe的参数
    scheduler = StepLR(optimizer=optimizer,step_size=15, gamma=0.8)
    criterion = nn.CrossEntropyLoss() # 
    
    
    weights_init(moe)
    moe.cuda()
    f = open(result_name, mode='w+')
    f.write("epoch,train_acc,train_wf1,test_acc,test_wf1,train_loss\n")
    f.close()
    # moe train
    print(f"-----------------------------training domain-sniff module-----------------------------")
    total_batches = 0
    for train_loader in train_loaders:
        total_batches += len(train_loader)
    for epoch in range(args.epochs):
        metrics.reset()
        moe.train()
        result = []
        total_loss = 0.0
        total_weights = torch.empty(0, len(models), requires_grad=False).cuda() # 所有权重
        for i in range(len(train_loaders)): 
            tqdm_train_loader = tqdm((train_loaders[i]), total=len(train_loaders[i]))
            for train_data in tqdm_train_loader:
                # 从原数据取出一个batch
                train_X, train_y, train_p = train_data
                train_X, train_y = train_X.float().cuda(), train_y.long().cuda()
                train_y = train_y.squeeze(1) # [X, 1] > [X,]
                optimizer.zero_grad()
                # 计算moe权重
                if int(args.equal) == 1:
                    weights = torch.tensor(np.array([1.0/(args.num_volunteers-1) for _ in range(args.num_volunteers-1)])).cuda().reshape(1,-1)
                else:
                    weights = moe(train_X)
                total_weights = torch.cat([total_weights, weights], dim=0) # 汇总权重

               
                # 计算moe合成结果
                moe_predict_y = torch.zeros(args.batch_size, args.num_classes).cuda()
                for j in range(len(models)):
                    if int(args.equal) == 0:
                        temp = weights[:, j].unsqueeze(1)
                    else:
                        temp = torch.ones(1,1).cuda()
                    with torch.no_grad():
                        model_predict_y = models[j](train_X) # 每个基模型对数据的预测
                    moe_predict_y += torch.mul(temp, model_predict_y)
                metrics.update(moe_predict_y, train_y)
                
                loss = criterion(moe_predict_y, train_y) 
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                tqdm_train_loader.set_description(f'Epoch [{epoch+1}/{args.epochs}] Loader [{i+1}/{len(train_loaders)}] ')
            # scheduler.step()
        
        accAndF1 = metrics.compute()
        
        result.extend([epoch+1, round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2)])
        print(f'Train: accuracy: {accAndF1["acc"].item() * 100:.2f}%, weighted_f1: {accAndF1["wf1"].item() * 100:.2f}%, loss: {total_loss / total_batches:.4f}')
        print(f"Train: weights: {np.round(torch.mean(total_weights, dim=0).detach().cpu().numpy(), 2)}")

        moe.eval()
        with torch.no_grad():
            metrics.reset()
            total_weights = torch.empty(0, len(models), requires_grad=False) # 所有权重
            for data in test_loader: 
                # 从原数据取出一个batch
                test_X, test_y, _ = data
                test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
                test_y = test_y.squeeze(1) # [X, 1] > [X,]
                if int(args.equal) == 1:
                    weights = torch.tensor(np.array([1.0/(args.num_volunteers-1) for _ in range(args.num_volunteers-1)])).cuda().reshape(1,-1)
                else:
                    weights = moe(train_X)
                total_weights = torch.cat([total_weights, weights.cpu()], dim=0) # 汇总权重          
                # 计算moe合成结果
                moe_predict_y = torch.zeros(args.batch_size, args.num_classes).cuda()
                for j in range(len(models)):
                    if int(args.equal) == 0:
                        temp = weights[:, j].unsqueeze(1)
                    else:
                        temp = torch.ones(1,1).cuda()
                    model_predict_y = models[j](test_X) # 每个基模型对数据的预测
                    moe_predict_y += torch.mul(temp, model_predict_y)
                metrics.update(moe_predict_y, test_y)
            
            accAndF1 = metrics.compute()
            print(f'Test: accuracy: {accAndF1["acc"] * 100:.2f}%, weighted_f1: {accAndF1["wf1"] * 100:.2f}%')
            print(f"Test: weights: {np.round(torch.mean(total_weights, dim=0).detach().cpu().numpy(), 2)}")
            result.extend([round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2), round(total_loss / total_batches , 4)])
            # 如果是最佳则保存
            if best_acc < accAndF1["acc"]:
                best_acc = accAndF1["acc"]
                best_wf1 = accAndF1["wf1"]
                best_models = models
                best_moe = moe
                best_metrics = accAndF1
                best_result = result
                best_weights = str(np.round(torch.mean(total_weights, dim=0).numpy(), 2).tolist())
            with open(result_name, mode='a') as f:
                f.write(str(result)[1:-1] + "\n")
    ckpts = {}
    for i in range(len(best_models)):
        ckpts.update({f"{args.model_name}_{i}": best_models[i].state_dict()})
    ckpts.update(best_metrics)
    ckpts.update({"moe":best_moe.state_dict(), "weight":best_weights})
    torch.save(ckpts, f"{result_path}/{args.model_name}_moe_{args.volunteer_split}_acc={best_acc * 100:.2f}_wf1={best_wf1 * 100:.2f}_seed={args.seed}.ckpt") 
    return best_result

def train_test_dann_with_torchmetrics(model, dataloader_source, dataloader_target, result_name, result_path, args):
    
    print(f"==============================================training {args.model_name}==============================================")
    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    model.cuda()
    metrics = get_metrics(args).cuda()
    
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()
    epochs = args.epochs
    weights_init(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0
    f = open(result_name, mode='w+')
    f.write("epoch,train_acc,train_wf1,test_acc,test_wf1,train_loss\n")
    f.close()
    for epoch in range(epochs):  # loop over the dataset multiple times
        metrics.reset()
        result = []
        total_loss = 0
        model.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
        for i in range(len_dataloader):  
            
            train_data = data_source_iter.__next__()      
            train_X, train_y, _ = train_data
            train_X, train_y = train_X.float().cuda(), train_y.long().cuda()
            train_y = train_y.squeeze(1) # [X, 1] > [X,]
            domain_label = torch.zeros(train_X.shape[0]).long().cuda()
            optimizer.zero_grad()
            class_output, domain_output = model(train_X)
            err_s_label = loss_class(class_output, train_y)
            err_s_domain = loss_domain(domain_output, domain_label)
            
            test_data = data_target_iter.__next__()
            test_X, _, _ = test_data
            test_X = test_X.float().cuda()
            
            domain_label = torch.ones(test_X.shape[0]).long().cuda()
            _, domain_output = model(test_X)
            err_t_domain = loss_domain(domain_output, domain_label)
            loss = err_t_domain + err_s_domain + err_s_label
            
            loss.backward()
            optimizer.step()    
            total_loss += loss.item()
            metrics.update(class_output, train_y)
   
        accAndF1 = metrics.compute()    
        print(f'Train: accuracy: {accAndF1["acc"].item() * 100:.2f}%, weighted_f1: {accAndF1["wf1"].item() * 100:.2f}%, loss: {total_loss / len_dataloader:.4f}')
        # Test time!
        result.extend([epoch+1, round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2)])
        model.eval()
        with torch.no_grad():       
            metrics.reset()
            for test_data in dataloader_target:              
                test_X, test_y, _  = test_data
                test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
                test_y = test_y.squeeze(1) # [X, 1] > [X,]
                
                class_output, _ = model(test_X) #  超大失误，test_X写成train_X
                metrics.update(class_output, test_y)
        # 本轮测试的所有结果 
        accAndF1 = metrics.compute()
        print(f'Test: accuracy: {accAndF1["acc"] * 100:.2f}%, weighted_f1: {accAndF1["wf1"] * 100:.2f}%')
        result.extend([round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2), "no loss here"])
        if True:
            best_acc = accAndF1["acc"]
            best_wf1 = accAndF1["wf1"]
            best_model = model
            best_metrics = accAndF1
            best_result = result
        with open(result_name, mode='a') as f:
            f.write(str(result)[1:-1] + "\n")
    torch.save({"model":best_model.state_dict()}.update(best_metrics), f"{result_path}/{args.model_name}_{args.volunteer_split}_acc={best_acc * 100:.2f}_wf1={best_wf1 * 100:.2f}_seed={args.seed}.ckpt") 
    return best_result

def train_test_deepcoral_with_torchmetrics(model:nn.Module, train_loader, test_loader, result_name, result_path, args):
    def CORAL(source, target):
        d = source.data.shape[1]

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss/(4*d*d)

        return loss
    print(f"==============================================training {args.model_name}==============================================")
    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    model = model.cuda()
    metrics = get_metrics(args).cuda()
   
    criterion = get_loss(args)
    optimizer = get_optimizer(model, args)
    total_batches = len(train_loader)
    epochs = args.epochs
    
    
    
    weights_init(model)
    scheduler = StepLR(optimizer=optimizer,step_size=20 )
    best_acc = 0.0
    f = open(result_name, mode='w+')
    f.write("epoch,train_acc,train_wf1,test_acc,test_wf1,train_loss\n")
    f.close()
    for epoch in range(epochs):  # loop over the dataset multiple times
        metrics.reset()
        result = []
        total_loss = 0
        model.train()
        total_batches = min(len(train_loader), len(test_loader))

        source = iter(train_loader)
        target = iter(test_loader)
        lanbuda = (epoch+1)/epochs
        for batch_index in range(total_batches):   
            train_data, test_data = source.__next__(), target.__next__()
                
            train_X, train_y, _ = train_data
            train_X, train_y = train_X.float().cuda(), train_y.long().cuda()
            train_y = train_y.squeeze(1) # [X, 1] > [X,]
            
            test_X, test_y, _  = test_data
            test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
            test_y = test_y.squeeze(1) # [X, 1] > [X,]
            
            optimizer.zero_grad()
            out1, out2 = model(train_X, test_X)
            loss = criterion(out1, train_y) + CORAL(out1, out2) * lanbuda
            loss.backward()
            optimizer.step()    
            total_loss += loss.item()
            metrics.update(out1, train_y)
            
        scheduler.step()    
        accAndF1 = metrics.compute()    
        print(f'Train: accuracy: {accAndF1["acc"].item() * 100:.2f}%, weighted_f1: {accAndF1["wf1"].item() * 100:.2f}%, loss: {total_loss / total_batches:.4f}')
        # Test time!
        result.extend([epoch+1, round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2)])
        model.eval()
        with torch.no_grad():       
            metrics.reset()
            for test_data in test_loader:              
                test_X, test_y, _  = test_data
                test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
                test_y = test_y.squeeze(1) # [X, 1] > [X,]
                
                predict_y, _ = model(test_X, test_X) #  超大失误，test_X写成train_X
                metrics.update(predict_y, test_y)
        # 本轮测试的所有结果 
        accAndF1 = metrics.compute()
        print(f'Test: accuracy: {accAndF1["acc"] * 100:.2f}%, weighted_f1: {accAndF1["wf1"] * 100:.2f}%')
        result.extend([round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2), round(total_loss / total_batches , 4)])
        if True:
            best_acc = accAndF1["acc"]
            best_wf1 = accAndF1["wf1"]
            best_model = model
            best_metrics = accAndF1
            best_result = result
        with open(result_name, mode='a') as f:
            f.write(str(result)[1:-1] + "\n")
    torch.save({"model":best_model.state_dict()}.update(best_metrics), f"{result_path}/{args.model_name}_{args.volunteer_split}_acc={best_acc * 100:.2f}_wf1={best_wf1 * 100:.2f}_seed={args.seed}.ckpt") 
    return best_result 

def train_test_rsc_with_torchmetrics(model:nn.Module, train_loader, test_loader, result_name, result_path, args):
    from torch import autograd
    import torch.nn.functional as F
    print(f"==============================================training {args.model_name}==============================================")
    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    def RSC(model, train_X, train_y, args):
        torch.cuda.set_device(f"cuda:{args.cuda_device}")
        all_p, all_f = model(train_X)
        # Equation (1): compute gradients with respect to representation
        all_o = torch.nn.functional.one_hot(train_y, args.num_classes)
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), model.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = model.fc(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), model.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        predict_y = model.fc(all_f * mask)
        return predict_y
    model = model.cuda()
    metrics = get_metrics(args).cuda()
    criterion = get_loss(args)
    optimizer = get_optimizer(model, args)
    total_batches = len(train_loader)
    epochs = args.epochs
    
    
    
    weights_init(model)
    scheduler = StepLR(optimizer=optimizer,step_size=20 )
    best_acc = 0.0
    f = open(result_name, mode='w+')
    f.write("epoch,train_acc,train_wf1,test_acc,test_wf1,train_loss\n")
    f.close()
    for epoch in range(epochs):  # loop over the dataset multiple times
        metrics.reset()
        result = []
        total_loss = 0
        model.train()
        tqdm_train_loader = tqdm((train_loader), total=(total_batches))
        for train_data in tqdm_train_loader:        
            train_X, train_y, train_p = train_data
            train_X, train_y, train_p = train_X.float().cuda(), train_y.long().cuda(), train_p.long().cuda()
            train_y = train_y.squeeze(1) # [X, 1] > [X,]
            train_p = train_p.squeeze(1) - 1 # 来自数据源的域标签是1开始
           

            optimizer.zero_grad()
            predict_y = RSC(model, train_X, train_y, args)
            loss = criterion(predict_y, train_y)
            loss.backward()
            optimizer.step()    
            total_loss += loss.item()
            metrics.update(predict_y, train_y)
            tqdm_train_loader.set_description(f'Epoch [{epoch+1}/{epochs}]')
            tqdm_train_loader.set_postfix_str(f'loss:{loss.item():.4f}')
        scheduler.step()    
        accAndF1 = metrics.compute()    
        print(f'Train: accuracy: {accAndF1["acc"].item() * 100:.2f}%, weighted_f1: {accAndF1["wf1"].item() * 100:.2f}%, loss: {total_loss / total_batches:.4f}')
        # Test time!
        result.extend([epoch+1, round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2)])
        model.eval()
        with torch.no_grad():       
            metrics.reset()
            for test_data in test_loader:              
                test_X, test_y, _  = test_data
                test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
                test_y = test_y.squeeze(1) # [X, 1] > [X,]
                
                predict_y, _ = model(test_X)
                metrics.update(predict_y, test_y)
        # 本轮测试的所有结果 
        accAndF1 = metrics.compute()
        print(f'Test: accuracy: {accAndF1["acc"] * 100:.2f}%, weighted_f1: {accAndF1["wf1"] * 100:.2f}%')
        result.extend([round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2), round(total_loss / total_batches , 4)])
        if True:
            best_acc = accAndF1["acc"]
            best_wf1 = accAndF1["wf1"]
            best_model = model
            best_metrics = accAndF1
            best_result = result
        with open(result_name, mode='a') as f:
            f.write(str(result)[1:-1] + "\n")
    torch.save({"model":best_model.state_dict()}.update(best_metrics), f"{result_path}/{args.model_name}_{args.volunteer_split}_acc={best_acc * 100:.2f}_wf1={best_wf1 * 100:.2f}_seed={args.seed}.ckpt") 
    return best_result

    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    results = []
    best_acc = 0.0
    best_result, best_weights = None, None
    expert_path = f"{result_path}/experts/{args.dataset_name}"
    import os 
    os.makedirs(expert_path, exist_ok=True)
    
    # train submodels
    for i in range(len(train_loaders)):
        model = get_model(args)
        trained_model = train(model, train_loaders[i], result_name,args)
        torch.save(trained_model, f"{expert_path}/{args.model_name}_{i}.pt")
        del trained_model
    # moe setup
    models = [torch.load(f"{expert_path}/{args.model_name}_{i}.pt") for i in range(len(train_loaders))]
    for i in range(len(models)):
        models[i] = models[i].eval()
    optimizer = Adam(params=moe.parameters(), lr=args.lr) # 一个优化器调整基模型+moe的参数
    scheduler = StepLR(optimizer=optimizer,step_size=15, gamma=0.8)
    criterion = nn.CrossEntropyLoss() # 
    
    
    weights_init(moe)
    moe.cuda()
    train_loss = [] # 每轮每个batch的loss合集，每轮的轮内平均loss
    # moe train
    print("=================moe=================")
    for epoch in range(args.epochs):
        moe.train()
        result = []
        print(f"Epoch #{epoch}:")
        total_y, total_y_pred_moe= [],[]
        train_epoch_loss = []
        total_weights = torch.empty(0, len(models), requires_grad=False).cuda() # 所有权重
        total_index = 0
        for i in range(len(train_loaders)): 
            
            for batch_index, train_data in enumerate(train_loaders[i]):
                # 假设一个batch里的志愿者标签一样
                total_index += 1 # 已经训练的batch数量
                # 从原数据取出一个batch
                train_X, train_y, train_p = train_data
                train_X, train_y = train_X.float().cuda(), train_y.long().cuda()
                train_y = train_y.squeeze(1) # [X, 1] > [X,]
                total_y.extend(train_y.cpu().numpy()) # 汇总真标签
                optimizer.zero_grad()

                # 计算moe权重
                if int(args.equal) == 1:
                    weights = torch.tensor(np.array([1.0/(args.num_volunteers-1) for _ in range(args.num_volunteers-1)])).cuda().reshape(1,-1)
                else:
                    weights = moe(train_X)
                total_weights = torch.cat([total_weights, weights], dim=0) # 汇总权重

               
                # 计算moe合成结果
                moe_predict_y = torch.zeros(args.batch_size, args.num_classes).cuda()
                for j in range(len(models)):
                    if int(args.equal) == 0:
                        temp = weights[:, j].unsqueeze(1)
                    else:
                        temp = torch.ones(1,1).cuda()
                    with torch.no_grad():
                        model_predict_y = models[j](train_X) # 每个基模型对数据的预测
                    moe_predict_y += torch.mul(temp, model_predict_y)
                total_y_pred_moe.extend(torch.max(moe_predict_y, dim=1)[1].cpu().numpy())
                
                loss = criterion(moe_predict_y, train_y) 
                loss.backward()
                train_epoch_loss.append(loss.item())
                train_loss.append(loss.item())
                optimizer.step()
            # scheduler.step()
            
        accuracy = accuracy_score(y_true=total_y, y_pred=total_y_pred_moe)
        f1 = f1_score(y_true=total_y, y_pred=total_y_pred_moe, average="macro")
        wf1 = f1_score(y_true=total_y, y_pred=total_y_pred_moe, average="weighted")
        recall = recall_score(y_true=total_y, y_pred=total_y_pred_moe, average="macro")
        precision = precision_score(y_true=total_y, y_pred=total_y_pred_moe, average="macro")
        result.extend([epoch+1,accuracy,recall,precision,f1,wf1])
        print(f"Train:loss={np.average(train_epoch_loss):>4f},accuracy={accuracy:.4f},f1={f1:.4f},wf1={wf1:.4f},recall={recall:.4f},precision={precision:.4f}")
        print(f"weights={torch.mean(total_weights, dim=0)}")

        moe.eval()
        loss_FF = get_loss(args)
        with torch.no_grad():
            total_y, total_y_pred_moe= [],[]
            test_epoch_loss = []
            total_weights = torch.empty(0, len(models), requires_grad=False) # 所有权重
            for batch_index, data in enumerate(test_loader): 
                # 从原数据取出一个batch
                test_X, test_y, _ = data
                test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
                test_y = test_y.squeeze(1) # [X, 1] > [X,]
                total_y.extend(test_y.cpu().numpy()) # 汇总真标签
                if int(args.equal) == 1:
                    weights = torch.tensor(np.array([1.0/(args.num_volunteers-1) for _ in range(args.num_volunteers-1)])).cuda().reshape(1,-1)
                else:
                    weights = moe(train_X)
                total_weights = torch.cat([total_weights, weights.cpu()], dim=0) # 汇总权重          
                # 计算moe合成结果
                moe_predict_y = torch.zeros(args.batch_size, args.num_classes).cuda()
                for j in range(len(models)):
                    if int(args.equal) == 0:
                        temp = weights[:, j].unsqueeze(1)
                    else:
                        temp = torch.ones(1,1).cuda()
                    model_predict_y = models[j](test_X) # 每个基模型对数据的预测
                    moe_predict_y += torch.mul(temp, model_predict_y)
                total_y_pred_moe.extend(torch.max(moe_predict_y, dim=1)[1].cpu().numpy())
                loss = loss_FF(moe_predict_y, test_y)
                test_epoch_loss.append(loss.item())
            accuracy = accuracy_score(y_true=total_y, y_pred=total_y_pred_moe)
            f1 = f1_score(y_true=total_y, y_pred=total_y_pred_moe, average="macro")
            wf1 = f1_score(y_true=total_y, y_pred=total_y_pred_moe, average="weighted")
            avg_loss = np.average(test_epoch_loss)
            recall = recall_score(y_true=total_y, y_pred=total_y_pred_moe, average="macro")
            precision = precision_score(y_true=total_y, y_pred=total_y_pred_moe, average="macro")
            result.extend([accuracy,recall,precision,f1,wf1])
            # 如果是最佳则保存
            if best_acc < accuracy:
                best_acc = accuracy
                best_result = result
                best_weights = str(torch.mean(total_weights, dim=0).tolist())
            print(f"Test:accuracy={accuracy:.4f},loss={avg_loss:.4f},f1={f1:.4f},wf1={wf1:.4f},recall={recall:.4f},precision={precision:.4f}")
            print(f"weights={torch.mean(total_weights, dim=0)}")
            results.append(result)
            np.savetxt(result_name,  np.array(results, dtype=np.float64), fmt='%.4f', delimiter=',')
    best_result.append(best_weights)
    return best_result


    
    print(f"==============================================training {args.model_name}==============================================")
    torch.cuda.set_device(f"cuda:{args.cuda_device}")
    model = model.cuda()
    metrics = get_metrics(args).cuda()
    # ============================================================
    if args.model_name == 'gile':
        false_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ============================================================
    criterion = get_loss(args)
    optimizer = get_optimizer(model, args)
    total_batches = len(train_loader)
    epochs = args.epochs
    
    
    
    weights_init(model)
    scheduler = StepLR(optimizer=optimizer,step_size=20 )
    best_acc = 0.0
    f = open(result_name, mode='w+')
    f.write("epoch,train_acc,train_wf1,test_acc,test_wf1,train_loss\n")
    f.close()
    for epoch in range(epochs):  # loop over the dataset multiple times
        metrics.reset()
        result = []
        total_loss = 0
        model.train()
        tqdm_train_loader = tqdm((train_loader), total=(total_batches))
        for i,train_data in enumerate(tqdm_train_loader):        
            train_X, train_y, train_p = train_data
            train_X, train_y, train_p = train_X.float().cuda(), train_y.long().cuda(), train_p.long().cuda()
            train_y = train_y.squeeze(1) # [X, 1] > [X,]
            train_p = train_p.squeeze(1) - 1 # 来自数据源的域标签是1开始
            # ============================================================
            if args.model_name == 'gile':
                optimizer.zero_grad()
                false_optimizer.zero_grad()
                loss, _ = model.loss_function(train_p, train_X, train_y)
                loss_false = model.loss_function_false(train_p, train_X, train_y)
                loss.backward()
                loss_false.backward()
                optimizer.step()
                false_optimizer.step()
                predict_y= model.classifier(train_X)[1] 
            # ============================================================
            else:
                optimizer.zero_grad()
                predict_y = model(train_X)
                x = model.get_feature()
                if epoch == 0 or epoch == 8:
                    if i < 5:
                        plot_tsne(x, train_y, f"{str(args.dataset_name).upper()}", epoch, i, args, train_p[0].item())
                loss = criterion(predict_y, train_y)
                loss.backward()
                optimizer.step()    
            total_loss += loss.item()
            metrics.update(predict_y, train_y)
            tqdm_train_loader.set_description(f'Epoch [{epoch+1}/{epochs}]')
            tqdm_train_loader.set_postfix_str(f'loss:{loss.item():.4f}')
        scheduler.step()    
        accAndF1 = metrics.compute()    
        print(f'Train: accuracy: {accAndF1["acc"].item() * 100:.2f}%, weighted_f1: {accAndF1["wf1"].item() * 100:.2f}%, loss: {total_loss / total_batches:.4f}')
        # Test time!
        result.extend([epoch+1, round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2)])
        model.eval()
        with torch.no_grad():       
            metrics.reset()
            for test_data in test_loader:              
                test_X, test_y, _  = test_data
                test_X, test_y = test_X.float().cuda(), test_y.long().cuda()
                test_y = test_y.squeeze(1) # [X, 1] > [X,]
                # ============================================================
                if args.model_name == "gile":
                    predict_y = model.classifier(test_X)[1]
                # ============================================================
                else:
                    predict_y = model(test_X) #  超大失误，test_X写成train_X
                metrics.update(predict_y, test_y)
        # 本轮测试的所有结果 
        accAndF1 = metrics.compute()
        print(f'Test: accuracy: {accAndF1["acc"] * 100:.2f}%, weighted_f1: {accAndF1["wf1"] * 100:.2f}%')
        result.extend([round(accAndF1['acc'].item() * 100,2),round(accAndF1['wf1'].item() * 100,2), round(total_loss / total_batches , 4)])
        if True:
            best_acc = accAndF1["acc"]
            best_wf1 = accAndF1["wf1"]
            best_model = model
            best_metrics = accAndF1
            best_result = result
        with open(result_name, mode='a') as f:
            f.write(str(result)[1:-1] + "\n")
    torch.save({"model":best_model.state_dict()}.update(best_metrics), f"{result_path}/{args.model_name}_{args.volunteer_split}_acc={best_acc * 100:.2f}_wf1={best_wf1 * 100:.2f}_seed={args.seed}.ckpt") 
    return best_result
if __name__ == '__main__':
    print("你运行的是utils.py!!!")
    
