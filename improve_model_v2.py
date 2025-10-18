#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Iyilestirme Scripti V2 - Daha Agresif Iyilestirmeler
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                            roc_auc_score, precision_recall_curve, roc_curve)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 MODEL IYILESTIRME V2 - DAHA AGRESIF IYILESTIRMELER")
print("="*80)

# ============================================================================
# 1. VERI YUKLEME VE DETAYLI ANALIZ
# ============================================================================
print("\n" + "="*80)
print("📊 ADIM 1: VERI YUKLEME VE DETAYLI ANALIZ")
print("="*80)

train_df = pd.read_csv('train_processed.csv')
test_df = pd.read_csv('test_processed.csv')

print(f"\n📊 Veri Boyutlari:")
print(f"   - Train: {train_df.shape}")
print(f"   - Test: {test_df.shape}")

# Boolean sutunlari numeric'e cevir
for col in train_df.columns:
    if train_df[col].dtype == bool:
        train_df[col] = train_df[col].astype(int)
    if col in test_df.columns and test_df[col].dtype == bool:
        test_df[col] = test_df[col].astype(int)

# Churn dagilimi
churn_dist = train_df['churn'].value_counts()
print(f"\n📈 Churn Dagilimi:")
for value, count in churn_dist.items():
    print(f"   Churn {value}: {count:,} ({count/len(train_df)*100:.2f}%)")

# ============================================================================
# 2. GELISMIS FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("🛠️ ADIM 2: GELISMIS FEATURE ENGINEERING")
print("="*80)

# Feature ve target ayirma
id_col = 'cust_id'
label_col = 'churn'
feature_cols = [c for c in train_df.columns if c not in [id_col, label_col]]

X = train_df[feature_cols].copy()
y = train_df[label_col].copy()
X_test = test_df[feature_cols].copy()

print(f"\n📊 Orijinal Feature Sayisi: {len(feature_cols)}")

# 1. Outlier temizleme (daha agresif)
print(f"\n🧹 Agresif Outlier Temizleme:")
numeric_cols = X.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    # Z-score yontemi ile outlier temizleme
    z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
    outlier_mask = z_scores > 3  # 3 standart sapma
    
    if outlier_mask.sum() > 0:
        # Outlier'lari median ile degistir
        median_val = X[col].median()
        X[col] = X[col].where(~outlier_mask, median_val)
        X_test[col] = X_test[col].where(~outlier_mask, median_val)
        print(f"   {col}: {outlier_mask.sum()} outlier temizlendi")

# 2. Feature transformation (PowerTransformer)
print(f"\n⚡ Feature Transformation (PowerTransformer):")
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_transformed = pt.fit_transform(X)
X_test_transformed = pt.transform(X_test)

X_transformed = pd.DataFrame(X_transformed, columns=feature_cols)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_cols)

print(f"   ✅ Power transformation tamamlandi!")

# 3. Feature selection (RFE ile)
print(f"\n🎯 RFE Feature Selection:")
# En iyi 30 feature'i sec
selector_rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=30)
selector_rfe.fit(X_transformed, y)

selected_features = X_transformed.columns[selector_rfe.support_].tolist()
print(f"   Secilen feature sayisi: {len(selected_features)}")

X_selected = X_transformed[selected_features]
X_test_selected = X_test_transformed[selected_features]

# ============================================================================
# 3. GELISMIS CLASS IMBALANCE COZUMU
# ============================================================================
print("\n" + "="*80)
print("⚖️ ADIM 3: GELISMIS CLASS IMBALANCE COZUMU")
print("="*80)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Split Sonrasi:")
print(f"   Train: {X_train.shape}")
print(f"   Validation: {X_val.shape}")

# Farkli sampling stratejileri dene
sampling_strategies = {
    'SMOTE': SMOTE(random_state=42, k_neighbors=3),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3),
    'ADASYN': ADASYN(random_state=42, n_neighbors=3),
    'SMOTETomek': SMOTETomek(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42)
}

best_sampling = None
best_f1 = 0

for name, sampler in sampling_strategies.items():
    print(f"\n🔄 {name} Test Ediliyor:")
    try:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"   Sonrası - Churn 0: {(y_resampled==0).sum():,}, Churn 1: {(y_resampled==1).sum():,}")
        
        # Hizli test (LightGBM ile)
        lgb_test = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        lgb_test.fit(X_resampled, y_resampled)
        y_pred_test = lgb_test.predict(X_val)
        f1_test = f1_score(y_val, y_pred_test)
        
        print(f"   F1-Score: {f1_test:.4f}")
        
        if f1_test > best_f1:
            best_f1 = f1_test
            best_sampling = sampler
            best_name = name
            
    except Exception as e:
        print(f"   Hata: {e}")

print(f"\n🏆 En Iyi Sampling: {best_name} (F1: {best_f1:.4f})")

# En iyi sampling ile final resampling
X_train_balanced, y_train_balanced = best_sampling.fit_resample(X_train, y_train)
print(f"   Final balanced - Churn 0: {(y_train_balanced==0).sum():,}, Churn 1: {(y_train_balanced==1).sum():,}")

# ============================================================================
# 4. GELISMIS MODEL EGITIMI VE HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("🤖 ADIM 4: GELISMIS MODEL EGITIMI VE HYPERPARAMETER TUNING")
print("="*80)

# Optimize edilmis hyperparameter'lar
models = {
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators': [300, 500, 700],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [0.1, 0.5]
        }
    },
    
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
        'params': {
            'n_estimators': [300, 500, 700],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [0.1, 0.5]
        }
    },
    
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    }
}

model_results = {}

for model_name, config in models.items():
    print(f"\n🚀 {model_name} Hyperparameter Tuning:")
    
    # GridSearchCV ile hyperparameter tuning
    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=3,  # 3-fold CV (hizli)
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    # En iyi model
    best_model = grid_search.best_estimator_
    
    # Validation prediction
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Metrikleri hesapla
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"   Best Params: {grid_search.best_params_}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    # Optimal threshold bulma
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
    
    # Optimal threshold ile prediction
    y_val_pred_optimal = (y_val_proba >= optimal_threshold).astype(int)
    f1_optimal = f1_score(y_val, y_val_pred_optimal)
    
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   F1-Score (optimal): {f1_optimal:.4f}")
    
    model_results[model_name] = {
        'model': best_model,
        'f1_score': f1,
        'f1_optimal': f1_optimal,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'predictions': y_val_pred_optimal,
        'probabilities': y_val_proba,
        'best_params': grid_search.best_params_
    }

# ============================================================================
# 5. ENSEMBLE MODEL
# ============================================================================
print("\n" + "="*80)
print("🎯 ADIM 5: ENSEMBLE MODEL")
print("="*80)

# En iyi 3 modeli sec
sorted_models = sorted(model_results.items(), key=lambda x: x[1]['f1_optimal'], reverse=True)
top_3_models = sorted_models[:3]

print(f"\n🏆 En Iyi 3 Model:")
for i, (model_name, results) in enumerate(top_3_models):
    print(f"   {i+1}. {model_name}: F1={results['f1_optimal']:.4f}")

# Ensemble model olustur
ensemble_estimators = []
for model_name, _ in top_3_models:
    ensemble_estimators.append((model_name, model_results[model_name]['model']))

# Voting weights
weights = [3, 2, 1]

ensemble_model = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft',
    weights=weights
)

print(f"\n🚀 Ensemble Model Egitimi:")
ensemble_model.fit(X_train_balanced, y_train_balanced)

# Ensemble validation
y_val_pred_ensemble = ensemble_model.predict(X_val)
y_val_proba_ensemble = ensemble_model.predict_proba(X_val)[:, 1]

f1_ensemble = f1_score(y_val, y_val_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_val, y_val_proba_ensemble)

# Optimal threshold
precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba_ensemble)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores[:-1])
optimal_threshold_ensemble = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5

y_val_pred_optimal_ensemble = (y_val_proba_ensemble >= optimal_threshold_ensemble).astype(int)
f1_optimal_ensemble = f1_score(y_val, y_val_pred_optimal_ensemble)

print(f"   F1-Score: {f1_ensemble:.4f}")
print(f"   F1-Score (optimal): {f1_optimal_ensemble:.4f}")
print(f"   ROC-AUC: {roc_auc_ensemble:.4f}")
print(f"   Optimal Threshold: {optimal_threshold_ensemble:.4f}")

# Ensemble sonuclarini ekle
model_results['Ensemble'] = {
    'model': ensemble_model,
    'f1_score': f1_ensemble,
    'f1_optimal': f1_optimal_ensemble,
    'roc_auc': roc_auc_ensemble,
    'optimal_threshold': optimal_threshold_ensemble,
    'predictions': y_val_pred_optimal_ensemble,
    'probabilities': y_val_proba_ensemble
}

# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("🔄 ADIM 6: CROSS-VALIDATION")
print("="*80)

# En iyi modeli sec
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1_optimal'])
best_model = model_results[best_model_name]['model']

print(f"\n🏆 En Iyi Model: {best_model_name}")

# 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_selected, y, cv=cv, scoring='f1', n_jobs=-1)

print(f"\n📊 Cross-Validation Sonuclari:")
print(f"   F1-Score (mean): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"   F1-Score (min): {cv_scores.min():.4f}")
print(f"   F1-Score (max): {cv_scores.max():.4f}")

# ============================================================================
# 7. FINAL MODEL VE PREDICTION
# ============================================================================
print("\n" + "="*80)
print("🎯 ADIM 7: FINAL MODEL VE PREDICTION")
print("="*80)

# Tüm train verisi ile final model egitimi
print(f"\n🚀 Final Model Egitimi (Tum Train Verisi):")
final_model = model_results[best_model_name]['model']

# Tüm train verisini dengele
X_train_final_balanced, y_train_final_balanced = best_sampling.fit_resample(X_selected, y)
final_model.fit(X_train_final_balanced, y_train_final_balanced)

print(f"   ✅ Final model egitimi tamamlandi!")

# Test prediction
optimal_threshold = model_results[best_model_name]['optimal_threshold']
y_test_proba = final_model.predict_proba(X_test_selected)[:, 1]
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

print(f"\n📊 Test Prediction:")
print(f"   Test ornek sayisi: {len(y_test_pred):,}")
print(f"   Predicted churn sayisi: {y_test_pred.sum():,}")
print(f"   Churn orani: {y_test_pred.sum()/len(y_test_pred)*100:.2f}%")

# Submission dosyasi olustur
submission = pd.DataFrame({
    'cust_id': test_df['cust_id'],
    'churn': y_test_pred
})
submission.to_csv('submission_improved_v2.csv', index=False)

print(f"\n✅ Submission dosyasi 'submission_improved_v2.csv' olarak kaydedildi!")

# Model kaydet
joblib.dump({
    'model': final_model,
    'transformer': pt,
    'selector': selected_features,
    'sampler': best_sampling,
    'threshold': optimal_threshold
}, 'best_model_improved_v2.pkl')

print(f"✅ Model 'best_model_improved_v2.pkl' olarak kaydedildi!")

# ============================================================================
# 8. SONUC RAPORU
# ============================================================================
print("\n" + "="*80)
print("📊 SONUC RAPORU")
print("="*80)

print(f"\n🎯 Model Performans Karsilastirmasi:")
print(f"{'Model':<15} {'F1-Score':<12} {'F1-Optimal':<12} {'ROC-AUC':<10} {'Threshold':<10}")
print("-" * 65)
for model_name, results in model_results.items():
    print(f"{model_name:<15} {results['f1_score']:<12.4f} {results['f1_optimal']:<12.4f} "
          f"{results['roc_auc']:<10.4f} {results['optimal_threshold']:<10.4f}")

print(f"\n🏆 En Iyi Model: {best_model_name}")
print(f"   - F1-Score: {model_results[best_model_name]['f1_score']:.4f}")
print(f"   - F1-Score (optimal): {model_results[best_model_name]['f1_optimal']:.4f}")
print(f"   - ROC-AUC: {model_results[best_model_name]['roc_auc']:.4f}")
print(f"   - Optimal Threshold: {model_results[best_model_name]['optimal_threshold']:.4f}")
print(f"   - CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n📊 Test Prediction Ozeti:")
print(f"   - Toplam musteri: {len(submission):,}")
print(f"   - Churn tahmin edilen: {submission['churn'].sum():,}")
print(f"   - Churn orani: {submission['churn'].sum()/len(submission)*100:.2f}%")

# Validation set detayli performans
best_predictions = model_results[best_model_name]['predictions']
print(f"\n📋 Validation Set Detayli Performans:")
print(classification_report(y_val, best_predictions, target_names=['No Churn', 'Churn']))

cm = confusion_matrix(y_val, best_predictions)
print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives: {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives: {cm[1,1]:,}")

print("\n" + "="*80)
print("🎉 MODEL IYILESTIRME V2 TAMAMLANDI!")
print("="*80)
