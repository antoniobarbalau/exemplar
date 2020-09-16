
import tensorflow as tf
import numpy as np

# @tf.function
def optimize(
    softmax,
    model,
    generator,
    start_image = None,
    population_size = 100,
    encoding_size = 10,
    elite_size = 25,
    max_iter = 100,
):

    encodings = tf.random.uniform(minval = -5, maxval = 5, shape = [population_size, encoding_size])
    softmax = tf.expand_dims(softmax, axis = 0)
    i = 0
    losses = tf.ones([population_size], dtype = tf.float32)
    while tf.reduce_mean(losses) > 0.30:
        texts = generator.generate(encodings)
        probabilities = model.predict(texts)
        print(f"========= ITERATION {i}", tf.reduce_mean(losses))

        print("#######################")
        for idx, text in enumerate(texts[:elite_size]):
            print(tf.round(1000 * probabilities[idx]) / 10, text)
        print("#######################")

        labels = tf.tile(softmax, [population_size, 1])
        losses = tf.keras.losses.mse(labels, probabilities)

        encodings = tf.gather_nd(
            encodings,
            tf.expand_dims(
                tf.argsort(losses),
                axis = -1
            )
        )
        encodings = tf.gather_nd(
            encodings,
            tf.expand_dims(
                tf.range(elite_size),
                axis = -1
            )
        )
        noise = tf.random.normal(stddev = 1, shape = (population_size, encoding_size))

        encodings = tf.tile(
            encodings,
            [population_size // elite_size, 1]
        )
        encodings = encodings + noise

        i += 1
        if i > max_iter:
            break

    texts = generator.generate(encodings)
    probabilities = model.predict(texts)
    return texts, probabilities, i
