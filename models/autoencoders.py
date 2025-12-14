import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers  # Nova importação necessária
import matplotlib.pyplot as plt
import seaborn as sns

class AutoencoderFraude:
    """
    Classe para criação e treinamento de um Autoencoder para detecção de fraudes.
    O modelo é treinado de forma semi-supervisionada (apenas em dados normais)
    e detecta anomalias baseado no erro de reconstrução.
    """
    
    def __init__(self, input_dim, encoding_dim=14):
        """
        Inicializa o Autoencoder.
        
        Args:
            input_dim (int): Número de features de entrada.
            encoding_dim (int): Dimensão da camada latente (gargalo).
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.history = None
        
    def construir_modelo(self):
        """Constrói a arquitetura do modelo (Encoder-Decoder)."""
        # Camada de Entrada
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        # Mudança: activation='elu' lida melhor com dados normalizados (negativos) como Sin/Cos
        encoded = Dense(22, activation='elu')(input_layer)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(18, activation='elu')(encoded)
        
        # Latent Space (Gargalo)
        # Mudança: Adição de regularização L1 para forçar esparsidade (bom para anomalias)
        encoded = Dense(self.encoding_dim, activation='elu',
                        activity_regularizer=regularizers.l1(10e-5))(encoded) 
        
        # Decoder
        decoded = Dense(18, activation='elu')(encoded)
        decoded = Dropout(0.1)(decoded)
        decoded = Dense(22, activation='elu')(decoded)
        
        # Saída linear pois usamos RobustScaler (dados não estão entre 0 e 1)
        decoded = Dense(self.input_dim, activation='linear')(decoded) 
        
        # Modelo Autoencoder Completo
        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        
        # Modelo apenas Encoder (para redução de dimensionalidade se necessário)
        self.encoder = Model(inputs=input_layer, outputs=encoded)
        
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        self.autoencoder.summary()
        
    def treinar(self, X_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Treina o modelo nos dados normais.
        
        Args:
            X_train (pd.DataFrame ou np.array): Dados de treino (apenas normais).
        """
        if self.autoencoder is None:
            self.construir_modelo()
            
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            mode='min'
        )
        
        self.history = self.autoencoder.fit(
            X_train, X_train, # Autoencoder: entrada = saída
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
    def plotar_historico(self):
        """Plota a curva de perda (loss) durante o treinamento."""
        if self.history is None:
            print("O modelo ainda não foi treinado.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Loss Treino')
        plt.plot(self.history.history['val_loss'], label='Loss Validação')
        plt.title('Progresso do Treinamento do Modelo (Loss MSE)')
        plt.ylabel('Erro Médio Quadrático (MSE)')
        plt.xlabel('Época')
        plt.legend()
        plt.grid()
        plt.show()
        
    def calcular_erro_reconstrucao(self, X):
        """
        Calcula o erro de reconstrução (MSE) para cada amostra.
        
        Args:
            X (pd.DataFrame): Dados para avaliar.
            
        Returns:
            np.array: Vetor com o MSE para cada amostra.
        """
        predictions = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        return mse
        
    def detectar_anomalias(self, X, threshold):
        """
        Classifica as amostras como normais (0) ou anomalias (1) baseada no threshold.
        """
        mse = self.calcular_erro_reconstrucao(X)
        y_pred = [1 if e > threshold else 0 for e in mse]
        return np.array(y_pred)
