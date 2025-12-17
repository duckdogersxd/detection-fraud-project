import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers, optimizers

class AutoencoderFraude:
    """
    Autoencoder robusto para detecção de anomalias em transações financeiras.
    Suporta arquitetura dinâmica, callbacks avançados e utilitários de análise.
    """
    
    def __init__(self, input_dim, encoding_dim=14, num_layers=2, 
                 neurons_decay=0.5, activation='elu', 
                 dropout_rate=0.1, l1_reg=10e-5, learning_rate=0.1):
        """
        Args:
            input_dim (int): Número de features de entrada.
            encoding_dim (int): Tamanho do gargalo (bottleneck).
            num_layers (int): Camadas ocultas no encoder (antes do gargalo).
            neurons_decay (float): Fator de redução de neurônios (0.0 a 1.0).
            activation (str): 'elu', 'relu', 'selu', 'tanh', 'swish'.
            dropout_rate (float): Taxa de dropout (0.0 a 0.5).
            l1_reg (float): Regularização L1 no gargalo (esparsidade).
            learning_rate (float): Taxa de aprendizado inicial.
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers
        self.neurons_decay = neurons_decay
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.learning_rate = learning_rate
        
        self.autoencoder = None
        self.history = None
        
    def _construir_camadas(self, input_tensor):
        """Lógica interna para empilhar camadas dinamicamente."""
        x = input_tensor
        
        # --- Encoder ---
        current_units = self.input_dim
        encoder_dims = [] 
        
        for i in range(self.num_layers):
            # Decaimento suave até chegar próximo do bottleneck
            current_units = max(int(current_units * self.neurons_decay), self.encoding_dim + 2)
            encoder_dims.append(current_units)
            
            x = Dense(current_units, activation=self.activation, name=f'enc_dense_{i}')(x)
            x = BatchNormalization(name=f'enc_bn_{i}')(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate, name=f'enc_drop_{i}')(x)
        
        # --- Bottleneck ---
        # L1 Regularization força representações latentes esparsas
        bottleneck = Dense(self.encoding_dim, activation=self.activation, 
                           activity_regularizer=regularizers.l1(self.l1_reg),
                           name='bottleneck')(x)
        
        # --- Decoder ---
        x = bottleneck
        for i, units in enumerate(reversed(encoder_dims)):
            x = Dense(units, activation=self.activation, name=f'dec_dense_{i}')(x)
            x = BatchNormalization(name=f'dec_bn_{i}')(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate, name=f'dec_drop_{i}')(x)
                
        # Saída Linear (Reconstrução precisa de valores reais como RobustScaler)
        output = Dense(self.input_dim, activation='linear', name='output')(x)
        return output

    def construir_modelo(self):
        """Monta e compila o modelo Keras."""
        input_layer = Input(shape=(self.input_dim,), name='input')
        output_layer = self._construir_camadas(input_layer)
        
        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        
        opt = optimizers.Adam(learning_rate=self.learning_rate)
        self.autoencoder.compile(optimizer=opt, loss='mse')
        
    def treinar(self, X_train, X_val=None, epochs=100, batch_size=64, verbose=0):
        """
        Treina o modelo com callbacks para evitar overfitting.
        """
        if self.autoencoder is None:
            self.construir_modelo()
            
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=verbose),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=verbose)
        ]
        
        if X_val is None:
            # Se não houver validação explícita, usa 20% do treino
            validation_split = 0.2
            validation_data = None
        else:
            validation_split = 0.0
            validation_data = (X_val, X_val)
            
        self.history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        return self.history

    def calcular_erro_reconstrucao(self, data):
        """Calcula Mean Squared Error (MSE) para cada amostra."""
        reconstructions = self.autoencoder.predict(data, verbose=0)
        mse = np.mean(np.power(data - reconstructions, 2), axis=1)
        return mse
        
    def detectar_anomalias(self, data, threshold):
        """Retorna previsões binárias (0=Normal, 1=Fraude) dado um threshold."""
        mse = self.calcular_erro_reconstrucao(data)
        return (mse > threshold).astype(int)

    def plotar_historico(self):
        """Plota curvas de perda de treino e validação."""
        if not self.history:
            print("Modelo ainda não treinado.")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Treino')
        plt.plot(self.history.history['val_loss'], label='Validação')
        plt.title('Histórico de Treinamento (Loss)')
        plt.ylabel('MSE')
        plt.xlabel('Época')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()