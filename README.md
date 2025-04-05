# 糖尿病神經病變預測專案
本專案旨在利用機器學習技術，透過患者的電生理數據以及多倫多量表數據預測糖尿病變（神經病變）的發生情形。  
專案主要以 Python 撰寫，並運用 pandas 與 scikit-learn 進行資料前處理、模型訓練與評估，同時支援對新資料進行預測。

## 專案內容
### 數據來源：
DSPN data collection_2020.12.19 前後兩筆(加嚴重程度分類) 共394人.xlsx：包含394位患者的電生理及多倫多數據與評估結果。  
1-1新資料用來check_R2_20211220_2.xlsx：新患者資料，用於檢查模型預測效能。  
神經病變測試結果.xlsx：模型預測結果（供參考與結果存檔）。

### 使用模型：
邏輯回歸 (Logistic Regression)  
決策樹 (Decision Tree)  
隨機森林 (Random Forest)

### 評估指標：
F1 Score、Accuracy、Precision、Sensitivity (Recall)、Specificity 及 AUC。

## 使用技術
程式語言： Python

### 主要套件：
pandas：數據讀取與處理  
scikit-learn：數據前處理、模型訓練與評估



## 安裝與執行
### 1. 環境準備
請確保您已安裝 Python 3.x。建議使用虛擬環境來管理專案依賴，並使用以下命令安裝所需套件：
pip install -r requirements.txt

### 2. 執行專案
在命令列中執行以下指令以運行程式，進行模型訓練與預測：
python src/預測患者是否有糖尿病變.py

程式會依序執行以下步驟：  
讀取原始數據並進行缺失值處理。  
分割訓練集與測試集，並建立多個機器學習模型。  
輸出各模型的評估結果（包括混淆矩陣、F1 Score、Accuracy、Precision、Sensitivity、Specificity 及 AUC）。  
使用訓練好的模型對新資料(1-1新資料用來check_R2_20211220_2.xlsx)進行預測，並在終端機中顯示預測結果（0 表示未罹患，1 表示已罹患）。

## 模型評估
專案中使用以下評估指標衡量模型效能：  
F1 Score： 綜合考慮 Precision 與 Recall 的表現。  
Accuracy： 模型預測的正確率。  
Precision： 預測為正類中實際為正類的比例。  
Sensitivity (Recall)： 真正例的識別率。  
Specificity： 真負例的識別率。  
AUC： ROC 曲線下的面積，衡量模型區分正負類能力。  
評估結果會在程式執行時輸出於終端機。

## 未來改進方向
數據處理： 探索其他缺失值處理與特徵工程方法，進一步提升預測效能。  
模型優化： 進行模型參數調整並嘗試其他機器學習或深度學習模型。  
自動化流程： 建立完整的數據處理、模型訓練與部署流水線，提升專案可擴展性。
