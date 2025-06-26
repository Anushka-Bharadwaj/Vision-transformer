from contextlib import nullcontext
import datetime
import glob
import json
import os
import time
import warnings
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



def compute_model_statistics(model, input_size=(3, 224, 224), device="cpu"):
    """
    Compute FLOPs, total parameters, estimated memory usage, and average inference time.
    Requires the 'ptflops' package.
    """
    try:
        from ptflops import get_model_complexity_info
        with torch.cuda.device(0) if device == "cuda" else nullcontext():
            flops, ptflops_params = get_model_complexity_info(
                model, input_size, as_strings=True,
                print_per_layer_stat=False, verbose=False
            )
    except ImportError:
        flops, ptflops_params = "N/A", "N/A"
        print("ptflops package not found. Skipping FLOPs calculation.")

    total_params = sum(p.numel() for p in model.parameters())
    estimated_memory_usage_bytes = total_params * 4  # assuming float32 (4 bytes per parameter)

    dummy_input = torch.randn(1, *input_size).to(device)
    model.eval()
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    # Measure inference time over 100 runs.
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 100

    stats = {
        "flops": flops,
        "ptflops_params": ptflops_params,
        "total_params": total_params,
        "estimated_memory_usage_bytes": estimated_memory_usage_bytes,
        "avg_inference_time_seconds": avg_inference_time
    }
    return stats



def calculate_metrics(y_true,y_pred,y_score=None):
    metrics={}
    metrics["accuracy_score"]=accuracy_score(y_pred,y_true)
    metrics["top_1_accuracy"]=metrics["accuracy_score"]
    if y_score is not None and y_score.shape[1]>=3:
        top3_correct=0
        for i,true_label in enumerate(y_true):
            top3_indices=np.argsort(y_score[i])[::-1][:3] 
            if true_label in top3_indices:
                top3_correct+=1
        metrics["top_3_accuracy"]=top3_correct/len(y_true)
            
    else:
        if y_score is not None and y_score.shape[1]<3:
            print("Less than 3 classes so top_3 accurcay will be same as top_1 accurcay")
            metrics["top_3_accuracy"]=metrics["top_1 accuracy"]
        else:
            metrics["top_3_accuracy"]=None
    # Precision
    metrics['precision_micro']=precision_score(y_true,y_pred,average='micro',zero_division=0)
    metrics['precision_macro']=precision_score(y_true,y_pred,average='macro',zero_division=0)
    metrics['precision_weighted']=precision_score(y_true,y_pred,average='weighted',zero_division=0)
        
    # Recall
    metrics['recall_micro']=recall_score(y_true,y_pred,average='micro',zero_division=0)
    metrics['recall_macro']=recall_score(y_true,y_pred,average='macro',zero_division=0)
    metrics['recall_weighted']=recall_score(y_true,y_pred,average='weighted',zero_division=0)
    
    # F1 Score
    metrics['f1_micro']=f1_score(y_true,y_pred,average='micro',zero_division=0)
    metrics['f1_macro']=f1_score(y_true,y_pred,average='macro',zero_division=0)
    metrics['f1_weighted']=f1_score(y_true,y_pred,average='weighted',zero_division=0)

    if y_score is not None:
        try:
            # One-hot encode the true labels for multi-class ROC AUC
            y_true_onehot = np.zeros((len(y_true),len(np.unique(y_true))))
            for i, val in enumerate(y_true):
                y_true_onehot[i,val]=1
            
            metrics['auc_micro']=roc_auc_score(y_true_onehot,y_score,average='micro',multi_class='ovr')
            metrics['auc_macro']=roc_auc_score(y_true_onehot,y_score,average='macro',multi_class='ovr')
            metrics['auc_weighted']=roc_auc_score(y_true_onehot,y_score,average='weighted',multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
            metrics['auc_micro']=metrics['auc_macro']=metrics['auc_weighted']=None
    return metrics


def find_latest_checkpoint(save_dir):
    checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint and return the starting epoch."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'] + 1  # Return next epoch to start from
    

dir="SoyMCData"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


train_dataset=datasets.ImageFolder(os.path.join(dir,"train"),transform=transform)
test_dataset=datasets.ImageFolder(os.path.join(dir,"test"),transform=transform)
val_dataset=datasets.ImageFolder(os.path.join(dir,"val"),transform=transform)


num_classes=len(train_dataset.classes)
num_classes



train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)


# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self,d_model,img_size,patch_size,n_channels):
        super().__init__()
        self.d_model=d_model
        self.img_size=img_size
        self.patch_size=patch_size
        self.n_channels=n_channels

        # converting the image into patches of data 
        self.layer=nn.Conv2d(in_channels=n_channels,out_channels=d_model,kernel_size=self.patch_size,stride=self.patch_size)
        
    def forward(self,x):
        x=self.layer(x)
        # Flattening the layer
        x=x.flatten(2)
        # Transposing the flatten version of the layer
        x=x.transpose(1,2)
        return x;
        


# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self,max_sequence_length,d_model):
        super().__init__()
        

# Class Token
        self.cls_token=nn.Parameter(torch.randn(1,1,d_model))
        
    
# Positional Embedding
        pe=torch.zeros(max_sequence_length,d_model)
        
        for pos in range(max_sequence_length):
          for i in range(d_model):
            if i % 2 == 0:
              pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
            else:
              pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))
                
        # since position embedding layer is fixed i.e not trainable so making it a buffer
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self,x):
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
    
        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch,x), dim=1)
    
        # Add positional embedding to embeddings
        x = x + self.pe
        return x;

# Self Attention
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim=768, key_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.key_dim = key_dim
        self.W = nn.Parameter(torch.randn(embedding_dim, 3 * key_dim))

    def forward(self, x):
        key_dim = self.key_dim
        qkv = torch.matmul(x, self.W)
        q = qkv[:, :, :key_dim]
        k = qkv[:, :, key_dim:key_dim*2]
        v = qkv[:, :, key_dim*2:]
        k_T = torch.transpose(k, -2, -1)
        dot_products = torch.matmul(q, k_T)
        scaled_dot_products = dot_products / np.sqrt(key_dim)
        attention_weights = F.softmax(scaled_dot_products, dim=-1)
        weighted_values = torch.matmul(attention_weights, v)
        return weighted_values



# MultiHead Self Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12):
        super().__init__()
        self.embeddding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0
        self.key_dim = embedding_dim // num_heads
        self.attention_list = [SelfAttention(embedding_dim, self.key_dim) for _ in range(num_heads)]
        self.multihead_attention = nn.ModuleList(self.attention_list)
        self.W = nn.Parameter(torch.randn(num_heads * self.key_dim, embedding_dim))

    def forward(self, x):
        attention_scores = [attention(x) for attention in self.multihead_attention]
        Z = torch.cat(attention_scores, -1)
        attention_score = torch.matmul(Z, self.W)
        return attention_score


# MLP
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=3072):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.mlp(x)




# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias




# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, hidden_dim=3072, dropout_prob=0.1):
        super().__init__()
        self.MSA = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.MLP = MultiLayerPerceptron(embedding_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        out_1 = self.dropout1(x)
        out_2 = self.layer_norm1(out_1)
        msa_out = self.MSA(out_2)
        out_3 = self.dropout2(msa_out)
        res_out = x + out_3
        out_4 = self.layer_norm2(res_out)
        mlp_out = self.MLP(out_4)
        out_5 = self.dropout3(mlp_out)
        output = res_out + out_5
        return output



# MLP Head
class MLPHead(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=10, fine_tune=False):
        super().__init__()
        if not fine_tune:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, 3072),
                nn.Tanh(),
                nn.Linear(3072, num_classes)
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.mlp_head(x)


#vision transformer
class Vision(nn.Module):
    def __init__(self,d_model,n_classes,img_size,patch_size,n_channels,n_heads,n_layers):
        super().__init__()

        self.d_model=d_model
        self.n_classes=n_classes
        self.img_size=img_size
        self.patch_size=patch_size
        self.n_channels=n_channels
        self.n_heads=n_heads
        self.n_layers=n_layers

        # calculating the number of patches
        self.n_patches=(self.img_size[0]*self.img_size[1])//(self.patch_size[0]*self.patch_size[1])
        
        # calculating the max sequence length 
        self.max_sequence_length=self.n_patches+1
        
        # Bring all layers together
        # Adding Patch Embedding Layer
        self.patch=PatchEmbedding(self.d_model,self.img_size,self.patch_size,self.n_channels)
        # Adding Positional Embedding Layer
        self.positional=PositionalEmbedding(self.max_sequence_length,self.d_model)
        # Adding Transformer Encoder Layer
        self.encoder=nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads) for _ in range(n_layers)])

        # Classification Head
        self.classification=nn.Sequential(
            nn.Linear(self.d_model,self.n_classes),
            nn.Softmax(dim=-1)
        )
        
    def forward(self,x):
        x=self.patch(x)
        x=self.positional(x)
        x=self.encoder(x)
        x=self.classification(x[:,0])
        return x;


d_model = 9
n_classes = 4
img_size = (224,224)
patch_size = (16,16)
n_channels = 3
n_heads = 3
n_layers = 3
epochs = 5
lr = 0.005
B = 1  # Batch size
N = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])  # Number of patches
D = d_model  # Embedding dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


model=Vision(d_model,n_classes,img_size,patch_size,n_channels,n_heads,n_layers)
model=model.to(device)


def train_eval(model,lr=1e-4,epochs=1,save_dir="./results"):
    # create dir if not exsists
    os.makedirs(save_dir,exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    #add learning rate scheduler
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5)
    
    train_losses, val_losses = [], []
    train_metrics_history = [] 
    val_metrics_history = []
    best_metrics = {
        'val_loss': float('inf'),
        'val_top1': 0.0,
        'val_top3': 0.0,
        'epoch': 0
    }
   
    #try to load the latest checkpoint
    latest_checkpoint=find_latest_checkpoint(save_dir)
    start_epoch=0
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
        print(f"Resuming training from epoch {start_epoch}")
        
        # Load metrics history if available
        metrics_file = os.path.join(save_dir, "results.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                results = json.load(f)
                train_losses = results.get("train_losses", [])
                val_losses = results.get("val_losses", [])
                train_metrics_history = results.get("train_metrics_history", [])
                val_metrics_history = results.get("val_metrics_history", [])
                
                # Load best metrics from history
                if val_metrics_history:
                    best_epoch_idx = min(range(len(val_metrics_history)), 
                                       key=lambda i: val_metrics_history[i].get("loss", float('inf')))
                    best_metrics = {
                        'val_loss': val_metrics_history[best_epoch_idx].get("loss", float('inf')),
                        'val_top1': val_metrics_history[best_epoch_idx].get("top1_accuracy", 0.0),
                        'val_top3': val_metrics_history[best_epoch_idx].get("top3_accuracy", 0.0),
                        'epoch': best_epoch_idx
                    }
                print(f"Loaded metrics history from previous training")
    
    
    training_start_time=time.time()
    
    for epoch in range(start_epoch,epochs):
        model.train()
        running_loss=0
        epoch_start_time=time.time()
        train_y_true, train_y_pred, train_y_score = [], [], []
        for inputs,labels in tqdm(train_loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            output=model(inputs)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            _,predicted=torch.max(output,1)
            train_y_true.extend(labels.cpu().numpy())
            train_y_pred.extend(predicted.cpu().numpy())
            train_y_score.extend(torch.softmax(output, dim=1).detach().cpu().numpy())
            
        training_loss=running_loss/len(train_loader)
        train_losses.append(training_loss)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(train_y_true, train_y_pred, np.array(train_y_score))
        train_metrics['loss'] = training_loss  # Add loss to the metrics dictionary
        train_metrics_history.append(train_metrics)

        model.eval()
        running_loss=0
        val_y_true, val_y_pred, val_y_score = [], [], []
        with torch.inference_mode():
            for inputs,labels in tqdm(val_loader):
                inputs=inputs.to(device)
                labels=labels.to(device)
                output=model(inputs)
                loss=criterion(output,labels)
                running_loss+=loss.item()
                _,predicted=torch.max(output,1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(predicted.cpu().numpy())
                val_y_score.extend(torch.softmax(output, dim=1).detach().cpu().numpy())
                
        val_loss=running_loss/len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_metrics=calculate_metrics(val_y_true,val_y_pred,np.array(val_y_score))
        val_metrics['loss'] = val_loss  # Add loss to the metrics dictionary
        val_metrics_history.append(val_metrics)


        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {training_loss:.4f}")
        print(f"  Train Top-1: {train_metrics['top_1_accuracy']*100:.2f}%")
        print(f"  Train Top-3: {train_metrics['top_3_accuracy']*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Top-1: {val_metrics['top_1_accuracy']*100:.2f}%")
        print(f"  Val Top-3: {val_metrics['top_3_accuracy']*100:.2f}%")

        scheduler.step(val_loss)

        if val_loss<best_metrics['val_loss']:
            best_metrics['val_loss']=val_loss
            best_metrics['val_top1']=val_metrics['top_1_accuracy']
            best_metrics['val_top3']=val_metrics['top_3_accuracy']
            checkpoint_path=os.path.join(save_dir,"best_model_checkpoint.pth")
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'loss':val_loss,
                'metrics':val_metrics,                
            },checkpoint_path)
            print(f"  Regular checkpoint saved for epoch {epoch+1}")
            
        if (epoch+1)%10==0:
            checkpoint_path=os.path.join(save_dir,f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'loss':val_loss,
                'metrics':val_metrics
            },checkpoint_path)
            print(f"Reguler Checkpoint saved for epoch {epoch+1}")
        # Save intermediate results after each epoch
        results = {
            "train_losses": [m['loss'] for m in train_metrics_history],
            "val_losses": [m['loss'] for m in val_metrics_history],
            "train_metrics_history": train_metrics_history,
            "val_metrics_history": val_metrics_history,
            "best_validation": {
                "epoch": best_metrics['epoch'] + 1,
                "metrics": val_metrics_history[best_metrics['epoch']] if best_metrics['epoch'] < len(val_metrics_history) else val_metrics_history[-1]
            }
        }
        results_path = os.path.join(save_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)    

    # Calculate total training time
    total_training_time=time.time()-training_start_time
    training_time_formatted = str(datetime.timedelta(seconds=int(total_training_time)))
    print(f"\nTotal training time: {training_time_formatted}")

    #Testing loop
    model.eval()
    test_loss=0
    test_y_true,test_y_pred,test_y_score=[],[],[]
    with torch.inference_mode():
        for inputs,labels in tqdm(test_loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            output=model(inputs)
            loss=criterion(output,labels)
            test_loss+=loss.item()
            _,predicted=torch.max(output,1)
            test_y_true.extend(labels.cpu().numpy())
            test_y_pred.extend(predicted.cpu().numpy())
            test_y_score.extend(torch.softmax(output,dim=1).detach().cpu().numpy())
    test_loss/=len(test_loader)
    test_metrics=calculate_metrics(test_y_true,test_y_pred,np.array(test_y_score))

    # Print final test results
    print("\nFinal Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Top-1: {test_metrics['top_1_accuracy']*100:.2f}%")
    print(f"  Top-3: {test_metrics['top_3_accuracy']*100:.2f}%")

    # Save all metrics history
    metrics_history = {
        'train': train_metrics_history,
        'val': val_metrics_history,
        'test': test_metrics,
        'best_validation': {
            'epoch': best_metrics['epoch'],
            'metrics': val_metrics_history[best_metrics['epoch']]
        }
    }

    #plot comparison btw top_1_acc and top_3_acc
    plt.figure(figsize=(10,6))
    # Plot comparison of top-1 and top-3 accuracies
    plt.figure(figsize=(10, 6))
    # Use the actual length of history instead of num_epochs
    actual_epochs = len(train_metrics_history)
    epochs = range(1, actual_epochs + 1)
    train_top1 = [m['top_1_accuracy'] * 100 for m in train_metrics_history]
    train_top3 = [m['top_3_accuracy'] * 100 for m in train_metrics_history]
    val_top1 = [m['top_1_accuracy'] * 100 for m in val_metrics_history]
    val_top3 = [m['top_3_accuracy'] * 100 for m in val_metrics_history]
    plt.plot(epochs, train_top1, 'b-', label='Train Top-1')
    plt.plot(epochs, train_top3, 'b--', label='Train Top-3')
    plt.plot(epochs, val_top1, 'r-', label='Val Top-1')
    plt.plot(epochs, val_top3, 'r--', label='Val Top-3')
    plt.title('Top-1 and Top-3 Accuracies Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'top1_top3_comparison.png'))
    plt.close()

    #plot loss curves
    plt.figure(figsize=(10,6))
    train_losses = [epoch_data['loss'] for epoch_data in train_metrics_history]
    val_losses = [epoch_data['loss'] for epoch_data in val_metrics_history]  
    plt.plot(epochs,train_losses,'b-',label='Train Loss')
    plt.plot(epochs,val_losses,'r-',label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir,'loss_curves.png'))
    plt.close()

    # Confusion matrix plot for test data
    conf_matrix = confusion_matrix(test_y_true, test_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    conf_matrix_path=os.path.join(save_dir, "test_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    
    # Confusion matrix plot for training data
    train_conf_matrix=confusion_matrix(train_y_true,train_y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
    plt.title("Training Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    train_conf_matrix_path=os.path.join(save_dir,"train_confusion_matrix.png")
    plt.savefig(train_conf_matrix_path)
    plt.close()

    # Save classification Reports
    train_y_true_all = []
    train_y_pred_all = []
    epochs=1
    for epoch in range(epochs):
        train_y_true_all.extend(train_metrics_history[epoch].get('y_true', []))
        train_y_pred_all.extend(train_metrics_history[epoch].get('y_pred', []))
    
    # If we don't have the raw predictions stored in metrics history, use the last epoch's data
    if not train_y_true_all:
        train_cls_report=classification_report(train_y_true,train_y_pred)
    else:
        train_cls_report=classification_report(train_y_true_all,train_y_pred_all)
    
    test_cls_report=classification_report(test_y_true,test_y_pred)
    
    train_report_path = os.path.join(save_dir,"train_classification_report.txt")
    with open(train_report_path,"w") as f:
        f.write(train_cls_report)
    
    test_report_path=os.path.join(save_dir,"test_classification_report.txt")
    with open(test_report_path,"w") as f:
        f.write(test_cls_report)

    # Save detailed metrics
    metrics_path = os.path.join(save_dir, "detailed_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("TRAINING METRICS (Final Epoch):\n")
        f.write("=============================\n")
        for metric, value in train_metrics_history[-1].items():
            if value is not None:
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: N/A\n")
        
        f.write("\nVALIDATION METRICS (Best Epoch):\n")
        f.write("==============================\n")
        best_val_metrics = val_metrics_history[best_metrics['epoch']]
        for metric, value in best_val_metrics.items():
            if value is not None:
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: N/A\n")
        
        f.write("\nTEST METRICS:\n")
        f.write("=============\n")
        for metric, value in test_metrics.items():
            if value is not None:
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: N/A\n")
        
        # Add training time information
        f.write("\nTRAINING TIME:\n")
        f.write("=============\n")
        f.write(f"Total training time: {training_time_formatted}\n")
        f.write(f"Average time per epoch: {total_training_time/epochs:.2f} seconds\n")

    # Save epoch wise data
    epoch_metric_path=os.path.join(save_dir,"training_metrics.txt")
    with open(epoch_metric_path,"w") as f:
        f.write("Epoch wise Training and Validation Metrics\n")
        for i, epoch_idx in enumerate(range(len(train_metrics_history))):
            actual_epoch = start_epoch + i  # Calculate the true epoch number
            f.write(f"Epoch {actual_epoch+1}:\n")
            f.write(f"  Train Loss: {train_metrics_history[epoch_idx]['loss']:.4f}\n")
            f.write(f"  Train Top-1: {train_metrics_history[epoch_idx]['top_1_accuracy']*100:.2f}%\n")
            f.write(f"  Train Top-3: {train_metrics_history[epoch_idx]['top_3_accuracy']*100:.2f}%\n")
            f.write(f"  Val Loss: {val_metrics_history[epoch_idx]['loss']:.4f}\n")
            f.write(f"  Val Top-1: {val_metrics_history[epoch_idx]['top_1_accuracy']*100:.2f}%\n")
            f.write(f"  Val Top-3: {val_metrics_history[epoch_idx]['top_3_accuracy']*100:.2f}%\n")
            f.write("\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Top-1: {test_metrics['top_1_accuracy']*100:.2f}%\n")
        f.write(f"Test Top-3: {test_metrics['top_3_accuracy']*100:.2f}%\n")
        f.write(f"Total training time: {training_time_formatted}\n")

    # Compute and Save model Statistics
    model_stats = compute_model_statistics(model, input_size=(3, 224, 224), device=device)
    stats_path = os.path.join(save_dir, "model_statistics.json")
    with open(stats_path, "w") as f:
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in list(obj)]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif obj is None:
                return None
            return obj
        
        json.dump(convert_to_serializable(model_stats), f, indent=4)

    # Compile all results in a dictionary and save to JSON
    results = {
        "train_losses": [m['loss'] for m in train_metrics_history],
        "val_losses": [m['loss'] for m in val_metrics_history],
        "train_top1_accuracies": train_top1,
        "train_top3_accuracies": train_top3,
        "val_top1_accuracies": val_top1,
        "val_top3_accuracies": val_top3,
        "test_loss": test_loss,
        "test_metrics": convert_to_serializable(test_metrics),
        "best_validation": {
            "epoch": best_metrics['epoch'] + 1,
            "metrics": convert_to_serializable(val_metrics_history[best_metrics['epoch']] if best_metrics['epoch'] < len(val_metrics_history) else val_metrics_history[-1])
        },
        "model_statistics": convert_to_serializable(model_stats),
        "training_time": {
            "total_seconds": total_training_time,
            "formatted": training_time_formatted,
            "average_epoch_seconds": total_training_time/epochs
        },
        "plots": {
            "top1_top3_comparison": os.path.join(save_dir, 'top1_top3_comparison.png'),
            "loss_curves": os.path.join(save_dir, 'loss_curves.png'),
            "test_confusion_matrix": conf_matrix_path,
            "train_confusion_matrix": train_conf_matrix_path
        }
    }
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save the trained model
    model_save_path = os.path.join(save_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
    print(f"All outputs have been saved to {os.path.abspath(save_dir)}")

    return model, metrics_history   



save_dir=r"C:\Users\Hp\CascadeProjects\vision_transformer\results"
train_eval(model,lr,epochs,save_dir)



summary(model)