import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):
        
        # Note: this is just an example.
        # Here the model.predict is called
        X = np.expand_dims(X, axis=2)
        out1 = self.model.predict(X)  # Shape [BSx9] for Phase 1 and [BSx18] for Phase 2
        out1 = np.expand_dims(out1, axis=-1)

        X_new = np.concatenate((X, out1), axis=1)
        X_final = X_new[:, -200:, :]

        out = self.model.predict(X_final)
        out = np.expand_dims(out, axis=-1)

        out_final = np.concatenate((out1, out), axis=1)
        out_final = out_final[:, :, 0]

        return out_final