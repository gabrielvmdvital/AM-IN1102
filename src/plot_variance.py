import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    MaxAbsScaler, 
    Normalizer
)
import os

def plot_all_variances():
    data_path = "data/spambase.data"
    if not os.path.exists(data_path):
        data_path = "../data/spambase.data"
        
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1].values
    
    scalers = {
        'Sem Normalização': None,
        'StandardScaler (Z-score)': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'Normalizer (Norma L2)': Normalizer()
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (name, scaler) in enumerate(scalers.items()):
        if scaler is None:
            X_trans = X
            color = 'royalblue'
        else:
            X_trans = scaler.fit_transform(X)
            color = 'seagreen'
            
        var_trans = np.var(X_trans, axis=0)
        
        axes[i].bar(range(len(var_trans)), var_trans, color=color, alpha=0.8)
        axes[i].set_title(f'Variância: {name}')
        axes[i].set_xlabel('Índice da Feature')
        axes[i].set_ylabel('Variância')
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)
        
        # Como vimos, a variância original é gigantesca, aplicamos escala log para visualizar
        if scaler is None:
            axes[i].set_yscale('log')
            axes[i].set_ylabel('Variância (escala log)')
            
    plt.tight_layout()
    plot_path = 'variance_all_scalers.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot salvo com sucesso em: {os.path.abspath(plot_path)}")

if __name__ == '__main__':
    plot_all_variances()
