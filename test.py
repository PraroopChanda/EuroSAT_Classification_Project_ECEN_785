import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import numpy as np



def run_model():

    ## setting the random seed
    _=torch.manual_seed(42)

    ''' Loading the data and creating the dataloader'''

    train_transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225])
    ])

    val_transforms=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225] 
        )])

    eurosat_dataset=datasets.EuroSAT(root="./datasets",download=True) ## give the path of the dataset

    ## 70% train and 30% validation split
    train_size=int(0.7*len(eurosat_dataset))
    val_size=int(0.15* len(eurosat_dataset))
    test_size=len(eurosat_dataset) -train_size -val_size

    train_dataset, val_dataset,test_dataset=random_split(eurosat_dataset,[train_size,val_size,test_size])
    ## applying transforms
    test_dataset.dataset.transform=val_transforms

    test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=4)

    #device --> use gpu if a CUDA enabled device
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Running on Device: {device}")


    ### Defining the Model
    class MultiClassSingleLabelModel(nn.Module):
        def __init__(self):
            super().__init__()
            backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            input_dim=backbone.fc.in_features # input features of the last fully connected layer of resnet which is 512
            backbone.fc=nn.Identity() # basically same feature input 512 dimensions is passed as an output instead of the 1000 (used for ImageNet classes)
            self.backbone=backbone
            self.head=nn.Sequential( ## new head for multi -label classification
                nn.Linear(input_dim,256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),  # improve generalization
                nn.Linear(256,10) # 10 is length of classes
            )

        def forward(self,x):
            feature=self.backbone(x)
            return self.head(feature)    

    ### Initializing the model loss and optimizer ###
    model=MultiClassSingleLabelModel().to(device)    

    ### loading the pre-trained model
    checkpoint=torch.load("./Final_ckpt_1e-5.pkt",map_location=device)
    model.load_state_dict(checkpoint,strict=True)

    print("---- Model Loaded Successfully ----")

    ## evaluation on Test Set
    def evaluation(model, test_loader, class_names=None):
        model.eval()
        all_preds = []
        all_labels = []
        test_loss=0
        criterion=nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                prd_class = model(images)

                loss=criterion(prd_class,labels)
                test_loss+=loss.item()

                preds = torch.argmax(prd_class, dim=1)
                all_preds.extend(preds.cpu().numpy()) ## adding all predictions into array
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss=test_loss/len(test_loader)
        # numpy conversion
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 (using macro which is ==> average over classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        print("------ Evaluation Metrics -------")
        print(f"Loss on Test Set: {avg_test_loss:.4f}")
        print(f"Accuracy on Test Set : {accuracy:.4f}")
        print(f"Precision on Test Set: {precision:.4f}")
        print(f"Recall on Test Set  : {recall:.4f}")
        print(f"F1-score on Test Set : {f1:.4f}")

        #plt.figure(figsize=(10, 8))
        disp.plot(cmap="Blues", xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        return accuracy, precision, recall, f1
    
    class_names=['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    evaluation(model,test_loader,class_names=class_names)

if __name__=='__main__':
    run_model()    




