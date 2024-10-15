import tensorflow as tf
from tensorflow.keras import layers, models

def build_vit_model(image_size, patch_size, num_classes):
    num_patches = (image_size // patch_size) ** 2
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Patch Embedding
    x = layers.Conv2D(filters=64, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    x = layers.Reshape((num_patches, -1))(x)

    # Positional Encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_encoding = layers.Embedding(input_dim=num_patches, output_dim=x.shape[-1])(positions)
    x = x + pos_encoding

    # Transformer Encoder Block
    def transformer_encoder(x, num_heads, ff_dim):
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)

        ff_output = layers.Dense(ff_dim, activation='relu')(x)
        ff_output = layers.Dense(x.shape[-1])(ff_output)
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
        return x

    # Add Transformer Layers
    x = transformer_encoder(x, num_heads=4, ff_dim=128)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create Model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model
