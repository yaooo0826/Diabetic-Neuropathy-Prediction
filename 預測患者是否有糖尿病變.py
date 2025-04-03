import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

# 讀取數據集
file = 'DSPN data collection_2020.12.19 前後兩筆(加嚴重程度分類) 共395人.xlsx'
data = pd.read_excel(file)

# 特徵與標籤
# 一對一
#---------------------------------------------------------------------------------------------------------------------------------------
# 電生理
X = data.loc[:, ['SNAP_Sur_L_1', 'SNCV_Sur_L_1', 'SLO_Sur_L_1', 'SNAP_Sur_R_1', 'SLO_Sur_R_1']]
# 多倫多量表
# X = data.loc[:, ['多倫多_ankle左_1','多倫多_ankle右_1','多倫多_振鈍右_1','多倫多_麻_1','多倫多_knee右_1']]  # CD 至 CY 欄位的特徵

y = data["確診神經病變_1"]  # B 欄標籤

# 一對二
#---------------------------------------------------------------------------------------------------------------------------------------
# 電生理
# X = data.loc[:, ['SNCV_Sur_L_1','SNAP_Sur_L_1','CMAP_Tib_R_1','SNAP_Ula_L_1','MNCV_Per_R_1']]  
# 多倫多量表
# X = data.loc[:, ['多倫多_ankle左_1','多倫多_刺_1','多倫多_knee左_1','多倫多_ankle右_1','多倫多_JPS鈍右_1']]

# y = data["確診神經病變_2"]  # CZ 欄標籤

#---------------------------------------------------------------------------------------------------------------------------------------
# 處理缺失值（移除 X 和 y 中任何包含缺失值的行）
data_cleaned = pd.concat([X, y], axis=1).dropna()
X = data_cleaned.iloc[:, :-1]  # 特徵
y = data_cleaned.iloc[:, -1]   # 標籤

# 分割數據為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 建立分類器
clf_lr = LogisticRegression(max_iter=500, random_state=53, n_jobs=-1)
clf_rf = RandomForestClassifier(max_depth=6, random_state=53, n_jobs=-1)
clf_dt = DecisionTreeClassifier(max_depth=4, random_state=53)

# 建立管道
scaler = StandardScaler()
pipe_lr = Pipeline([('scaler', scaler), ('clf', clf_lr)])
pipe_rf = Pipeline([('clf', clf_rf)])
pipe_dt = Pipeline([('clf', clf_dt)])

models = {
    'Logistic Regression': pipe_lr,
    'Decision Tree': pipe_dt,
    'Random Forest': pipe_rf
}

# 評估函數（以文字輸出模糊矩陣）
def evaluate_model(y_true, y_pred, y_prob=None, model_name=None):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    sensitivity = TP / (TP + FN)  # Recall / TPR
    specificity = TN / (TN + FP)  # TNR
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')

    auc = roc_auc_score(y_true, y_prob, multi_class='ovr') if y_prob is not None else None

    print(f"----------{model_name}----------")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    if auc is not None:
        print(f"AUC: {auc:.2f}")
    print("\nConfusion Matrix:")
    print(f"[[{cm[0, 0]} {cm[0, 1]}]\n [{cm[1, 0]} {cm[1, 1]}]]")
    print("\n")

# 模型訓練與評估
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 如果支持概率預測，則計算 AUC
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    evaluate_model(y_test, y_pred, y_prob, model_name=model_name)

#-----------------------------------------------------------------------------------------------------------------------------------------------

# 載入新資料集 (只有四位病人的特徵, 無標籤)
new_file = '1-1新資料用來check_R2_20211220_2.xlsx'
new_data = pd.read_excel(new_file)

# 確保新資料的欄位順序與訓練資料時所使用的特徵一致
feature_cols = X_train.columns  
new_data = new_data[feature_cols]

# 如果新資料有缺失值，考慮以同樣方式處理，如 dropna() 或補值
new_data = new_data.dropna()


# 使用已訓練好的模型進行預測
models = {
    'Logistic Regression': pipe_lr,
    'Decision Tree': pipe_dt,
    'Random Forest': pipe_rf
}

for model_name, model in models.items():
    predictions = model.predict(new_data)
    # 假設: 1 = 得病, 0 = 未得病 (此標籤定義需與原訓練資料一致)
    print(f"----{model_name} Predictions----")
    for i, pred in enumerate(predictions, start=1):
        result = "1" if pred == 1 else "0"
        print(f"患者 {i}: {result}")
    print("\n")

