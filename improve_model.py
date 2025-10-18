#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Iyilestirme Scripti
Tum iyilestirme adimlarini sirasiyla uygular
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                            roc_auc_score, precision_recall_curve, roc_curve)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 MODEL İYİLEŞTİRME SÜRECİ BAŞLIYOR")
print("="*80)

# ============================================================================
# 1. VERİ YÜKLEME VE İLK ANALİZ
# ============================================================================
print("\n" + "="*80)
print("📊 ADIM 1: VERİ YÜKLEME VE İLK ANALİZ")
print("="*80)

train_df = pd.read_csv('train_processed.csv')
test_df = pd.read_csv('test_processed.csv')

print(f"\n📊 Veri Boyutları:")
print(f"   - Train: {train_df.shape}")
print(f"   - Test: {test_df.shape}")

# Boolean sütunları numeric'e çevir
for col in train_df.columns:
    if train_df[col].dtype == bool:
        train_df[col] = train_df[col].astype(int)
    if col in test_df.columns and test_df[col].dtype == bool:
        test_df[col] = test_df[col].astype(int)

# Eksik değer analizi
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print(f"\n📋 Eksik Değer Analizi:")
print(f"   Train toplam eksik: {missing_train.sum()}")
print(f"   Test toplam eksik: {missing_test.sum()}")

# Churn dağılımı
churn_dist = train_df['churn'].value_counts()
print(f"\n📈 Churn Dağılımı:")
for value, count in churn_dist.items():
    print(f"   Churn {value}: {count:,} ({count/len(train_df)*100:.2f}%)")

imbalance_ratio = churn_dist[0] / churn_dist[1]
print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")

# ============================================================================
# 2. FEATURE ENGINEERING İYİLEŞTİRME
# ============================================================================
print("\n" + "="*80)
print("🛠️ ADIM 2: FEATURE ENGINEERING İYİLEŞTİRME")
print("="*80)

# Feature ve target ayırma
id_col = 'cust_id'
label_col = 'churn'
feature_cols = [c for c in train_df.columns if c not in [id_col, label_col]]

X = train_df[feature_cols].copy()
y = train_df[label_col].copy()
X_test = test_df[feature_cols].copy()

print(f"\n📊 Feature Sayısı: {len(feature_cols)}")

# Outlier temizleme (IQR yöntemi)
print(f"\n🧹 Outlier Temizleme:")
numeric_cols = X.select_dtypes(include=[np.number]).columns

outlier_counts = {}
for col in numeric_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # 3 IQR kullan (daha az agresif)
    upper_bound = Q3 + 3 * IQR
    
    # Outlier'ları cap et (silme yerine)
    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
    X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)
    
    outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
    if outliers > 0:
        outlier_counts[col] = outliers

print(f"   Outlier temizlenen sütun sayısı: {len(outlier_counts)}")

# Feature scaling (RobustScaler - outlier'lara karşı daha dayanıklı)
print(f"\n⚖️ Feature Scaling (RobustScaler):")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

print(f"   ✅ Scaling tamamlandı!")

# Feature selection (mutual information)
print(f"\n🎯 Feature Selection (Mutual Information):")
selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(X_scaled, y)

# Feature importance skorları
feature_scores = pd.DataFrame({
    'feature': feature_cols,
    'score': selector.scores_
}).sort_values('score', ascending=False)

# En iyi K feature'ı seç (top 80%)
k_best = int(len(feature_cols) * 0.8)
top_features = feature_scores.head(k_best)['feature'].tolist()

print(f"   Orijinal feature sayısı: {len(feature_cols)}")
print(f"   Seçilen feature sayısı: {len(top_features)}")
print(f"   Top 10 en önemli feature:")
for i, (_, row) in enumerate(feature_scores.head(10).iterrows()):
    print(f"      {i+1:2d}. {row['feature']}: {row['score']:.4f}")

X_selected = X_scaled[top_features]
X_test_selected = X_test_scaled[top_features]

# ============================================================================
# 3. CLASS IMBALANCE ÇÖZÜMÜ
# ============================================================================
print("\n" + "="*80)
print("⚖️ ADIM 3: CLASS IMBALANCE ÇÖZÜMÜ")
print("="*80)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Split Sonrası:")
print(f"   Train: {X_train.shape}")
print(f"   Validation: {X_val.shape}")

# SMOTE uygula
print(f"\n🔄 SMOTE Uygulanıyor:")
print(f"   Öncesi - Churn 0: {(y_train==0).sum():,}, Churn 1: {(y_train==1).sum():,}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   Sonrası - Churn 0: {(y_train_balanced==0).sum():,}, Churn 1: {(y_train_balanced==1).sum():,}")
print(f"   ✅ SMOTE tamamlandı!")

# ============================================================================
# 4. MODEL EĞİTİMİ VE HYPERPARAMETREoptimizasyonu
# ============================================================================
print("\n" + "="*80)
print("🤖 ADIM 4: MODEL EĞİTİMİ VE OPTIMIZASYON")
print("="*80)

# Model tanımlamaları (optimize edilmiş hyperparameter'lar)
models = {
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    
    'XGBoost': xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ),
    
    'CatBoost': CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        thread_count=-1
    )
}

model_results = {}

for model_name, model in models.items():
    print(f"\n🚀 {model_name} Eğitimi:")
    
    # Model eğitimi
    model.fit(X_train_balanced, y_train_balanced)
    
    # Validation prediction
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Metrikleri hesapla
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    # Optimal threshold bulma
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    
    # Optimal threshold ile prediction
    y_val_pred_optimal = (y_val_proba >= optimal_threshold).astype(int)
    f1_optimal = f1_score(y_val, y_val_pred_optimal)
    
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   F1-Score (optimal): {f1_optimal:.4f}")
    
    model_results[model_name] = {
        'model': model,
        'f1_score': f1,
        'f1_optimal': f1_optimal,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'predictions': y_val_pred_optimal,
        'probabilities': y_val_proba
    }

# ============================================================================
# 5. CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("🔄 ADIM 5: CROSS-VALIDATION")
print("="*80)

# En iyi modeli seç
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1_optimal'])
best_model = model_results[best_model_name]['model']

print(f"\n🏆 En İyi Model: {best_model_name}")

# 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_selected, y, cv=cv, scoring='f1', n_jobs=-1)

print(f"\n📊 Cross-Validation Sonuçları:")
print(f"   F1-Score (mean): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"   F1-Score (min): {cv_scores.min():.4f}")
print(f"   F1-Score (max): {cv_scores.max():.4f}")

# ============================================================================
# 6. FINAL MODEL VE PREDICTION
# ============================================================================
print("\n" + "="*80)
print("🎯 ADIM 6: FINAL MODEL VE PREDICTION")
print("="*80)

# Tüm train verisi ile final model eğitimi
print(f"\n🚀 Final Model Eğitimi (Tüm Train Verisi):")
final_model = models[best_model_name]

# SMOTE ile tüm train verisini dengele
X_train_final_balanced, y_train_final_balanced = smote.fit_resample(X_selected, y)
final_model.fit(X_train_final_balanced, y_train_final_balanced)

print(f"   ✅ Final model eğitimi tamamlandı!")

# Test prediction
optimal_threshold = model_results[best_model_name]['optimal_threshold']
y_test_proba = final_model.predict_proba(X_test_selected)[:, 1]
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

print(f"\n📊 Test Prediction:")
print(f"   Test örnek sayısı: {len(y_test_pred):,}")
print(f"   Predicted churn sayısı: {y_test_pred.sum():,}")
print(f"   Churn oranı: {y_test_pred.sum()/len(y_test_pred)*100:.2f}%")

# Submission dosyası oluştur
submission = pd.DataFrame({
    'cust_id': test_df['cust_id'],
    'churn': y_test_pred
})
submission.to_csv('submission_improved.csv', index=False)

print(f"\n✅ Submission dosyası 'submission_improved.csv' olarak kaydedildi!")

# Model kaydet
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'selector': top_features,
    'threshold': optimal_threshold
}, 'best_model_improved.pkl')

print(f"✅ Model 'best_model_improved.pkl' olarak kaydedildi!")

# ============================================================================
# 7. SONUÇ RAPORU
# ============================================================================
print("\n" + "="*80)
print("📊 SONUÇ RAPORU")
print("="*80)

print(f"\n🎯 Model Performans Karşılaştırması:")
print(f"{'Model':<15} {'F1-Score':<12} {'F1-Optimal':<12} {'ROC-AUC':<10} {'Threshold':<10}")
print("-" * 65)
for model_name, results in model_results.items():
    print(f"{model_name:<15} {results['f1_score']:<12.4f} {results['f1_optimal']:<12.4f} "
          f"{results['roc_auc']:<10.4f} {results['optimal_threshold']:<10.4f}")

print(f"\n🏆 En İyi Model: {best_model_name}")
print(f"   - F1-Score: {model_results[best_model_name]['f1_score']:.4f}")
print(f"   - F1-Score (optimal): {model_results[best_model_name]['f1_optimal']:.4f}")
print(f"   - ROC-AUC: {model_results[best_model_name]['roc_auc']:.4f}")
print(f"   - Optimal Threshold: {model_results[best_model_name]['optimal_threshold']:.4f}")
print(f"   - CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n📊 Test Prediction Özeti:")
print(f"   - Toplam müşteri: {len(submission):,}")
print(f"   - Churn tahmin edilen: {submission['churn'].sum():,}")
print(f"   - Churn oranı: {submission['churn'].sum()/len(submission)*100:.2f}%")

# Validation set detaylı performans
best_predictions = model_results[best_model_name]['predictions']
print(f"\n📋 Validation Set Detaylı Performans:")
print(classification_report(y_val, best_predictions, target_names=['No Churn', 'Churn']))

cm = confusion_matrix(y_val, best_predictions)
print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives: {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives: {cm[1,1]:,}")

print("\n" + "="*80)
print("🎉 MODEL İYİLEŞTİRME SÜRECİ TAMAMLANDI!")
print("="*80)

